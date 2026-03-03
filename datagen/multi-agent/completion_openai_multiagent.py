import os
import argparse
import json
import asyncio
from typing import Dict, Any, List
from openai import AsyncClient
from agents import Agent, OpenAIResponsesModel, Runner, SQLiteSession
from virtual_tools import VirtualToolBackend, create_dynamic_virtual_tool
from utils import load_dataset_from_file

# Configuration
def get_args():
    parser = argparse.ArgumentParser(description="Multi-Agent Simulation Manager")
    parser.add_argument("--input_file", type=str, required=True, help="Input prepared dataset (.jsonl)")
    
    # Model Configs
    parser.add_argument("--model_path", type=str, default="z-ai/glm-4.7", help="Student Model")
    parser.add_argument("--teacher_model", type=str, default="moonshotai/kimi-k2-thinking", help="Teacher (Proctor) Model")
    parser.add_argument("--answer_gen_model", type=str, default="moonshotai/kimi-k2-thinking", help="Oracle Model")
    parser.add_argument("--virtual_tool_model", type=str, default="z-ai/glm-4.7", help="Model simulating the tools")
    
    # API Configs
    parser.add_argument("--openrouter_api_key", type=str, default=os.getenv("OPENROUTER_API_KEY"))
    parser.add_argument("--openrouter_url", type=str, default="https://openrouter.ai/api/v1")
    
    # Run Configs
    parser.add_argument("--max_turns", type=int, default=1)
    parser.add_argument("--virtual_tools", action="store_true", default=True)
    return parser.parse_args()

def sanitize_tool_schema(tool_def: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively removes 'additionalProperties'."""
    def _clean(obj):
        if isinstance(obj, dict):
            obj.pop('additionalProperties', None)
            for k, v in obj.items():
                _clean(v)
        elif isinstance(obj, list):
            for item in obj:
                _clean(item)
    
    clean_def = json.loads(json.dumps(tool_def))
    if 'input_schema' in clean_def:
        _clean(clean_def['input_schema'])
    return clean_def

def format_ground_truth_for_teacher(ground_truth_outputs: List[Dict]) -> str:
    """Extracts ONLY the 'response' content for the Teacher."""
    clean_answers = []
    for entry in ground_truth_outputs:
        output_block = entry.get('output', {})
        if isinstance(output_block, dict) and 'response' in output_block:
            clean_answers.append(output_block['response'])
        else:
            clean_entry = output_block.copy() if isinstance(output_block, dict) else output_block
            if isinstance(clean_entry, dict):
                clean_entry.pop('error', None)
            clean_answers.append(clean_entry)
    return json.dumps(clean_answers, indent=2)

def parse_agent_result(result) -> List[Dict[str, Any]]:
    """
    The 'Toucan Approach': Manually iterates through the agent's new items
    and constructs clean, standard dictionary messages.
    """
    new_messages = []
    
    if hasattr(result, 'new_items') and result.new_items:
        current_reasoning = []  # Buffer for reasoning text
        
        for item_flow in result.new_items:
            
            # --- 1. REASONING ---
            if item_flow.type == "reasoning_item":
                if hasattr(item_flow, 'raw_item') and hasattr(item_flow.raw_item, 'content'):
                    for content in item_flow.raw_item.content:
                        if hasattr(content, 'text'):
                            current_reasoning.append(content.text)
            
            # --- 2. TOOL CALLS ---
            elif item_flow.type == "tool_call_item":
                # Flush reasoning buffer first
                if current_reasoning:
                    new_messages.append({
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "\n".join(current_reasoning),
                        "type": "reasoning"
                    })
                    current_reasoning = []
                
                # Extract tool details
                if hasattr(item_flow, 'raw_item'):
                    tool_call = {
                        "name": getattr(item_flow.raw_item, 'name', None),
                        "arguments": getattr(item_flow.raw_item, 'arguments', None),
                        "call_id": getattr(item_flow.raw_item, 'call_id', None)
                    }
                    
                    new_messages.append({
                        "role": "assistant",
                        "content": "",
                        "function_call": tool_call,
                        "type": "function_call"
                    })
            
            # --- 3. TOOL OUTPUTS ---
            elif item_flow.type == "tool_call_output_item":
                # Extract output
                tool_output = item_flow.output
                # Attempt to parse inner JSON if wrapped
                try:
                    output_data = json.loads(tool_output)
                    if isinstance(output_data, dict) and output_data.get('type') == 'text':
                        inner_data = json.loads(output_data.get('text', '{}'))
                        tool_output = json.dumps(inner_data)
                except:
                    pass
                
                # Try to find tool name (Toucan logic looks back, we use best effort)
                tool_name = 'unknown'
                if hasattr(item_flow, 'raw_item'):
                    # In a runner stream, the raw item often links back to the call
                    pass 

                new_messages.append({
                    "role": "function",
                    "content": tool_output,
                    "name": tool_name
                })

            # --- 4. TEXT MESSAGES ---
            elif item_flow.type == "message_output_item":
                if hasattr(item_flow, 'raw_item') and hasattr(item_flow.raw_item, 'content'):
                    message_texts = []
                    for content in item_flow.raw_item.content:
                        if hasattr(content, 'text'):
                            message_texts.append(content.text)
                    
                    # Flush reasoning buffer first
                    if current_reasoning:
                        new_messages.append({
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "\n".join(current_reasoning),
                            "type": "reasoning"
                        })
                        current_reasoning = []
                    
                    final_content = "\n".join(message_texts)
                    if final_content.strip():
                        new_messages.append({
                            "role": "assistant",
                            "content": final_content,
                            "type": "message"
                        })

        # --- FINAL FLUSH ---
        # If there's leftover reasoning at the very end
        if current_reasoning:
             new_messages.append({
                "role": "assistant",
                "content": "",
                "reasoning_content": "\n".join(current_reasoning),
                "type": "reasoning"
            })
            
    # Fallback: If no complex items, just grab the final output string
    if not new_messages and result.final_output:
        new_messages.append({
            "role": "assistant",
            "content": result.final_output,
            "type": "message"
        })

    return new_messages

async def run_student_teacher_loop(
    student_agent: Agent,
    teacher_agent: Agent,
    row_id: str,
    initial_query: str,
    max_turns: int = 10
) -> List[Dict[str, Any]]:
    
    student_session = SQLiteSession(f"session_student_{row_id}")
    await student_session.clear_session()
    teacher_session = SQLiteSession(f"session_teacher_{row_id}")
    await teacher_session.clear_session()

    print(f"   🏁 [Start] Initial Query: {initial_query[:100]}...")
    
    # Pre-fill Teacher Memory
    await teacher_session.add_items([{
        "role": "assistant", 
        "content": initial_query
    }])
    
    # Init History
    full_history = []
    full_history.append({"role": "user", "content": initial_query})
    
    next_student_input = initial_query
    
    for turn in range(max_turns):
        print(f"   🔄 Turn {turn + 1}/{max_turns}")

        # --- STUDENT TURN ---
        try:
            student_result = await Runner.run(
                student_agent, 
                input=next_student_input, 
                session=student_session
            )
        except Exception as e:
            print(f"   ❌ Student Agent Error: {e}")
            full_history.append({"role": "system", "content": f"Student Error: {e}"})
            break
        
        # --- TOUCAN CLEANING ---
        # Parse the result into clean dictionaries
        new_msgs = parse_agent_result(student_result)
        full_history.extend(new_msgs)
        
        # Get text for Teacher (grab the last message's content)
        student_response_text = student_result.final_output
        print(f"   🤖 [Student]: {student_response_text[:100]}...")
        
        # --- TEACHER TURN ---
        teacher_input = f"The Assistant responded: {student_response_text}"
        
        try:
            teacher_result = await Runner.run(
                teacher_agent, 
                input=teacher_input, 
                session=teacher_session
            )
            teacher_feedback = teacher_result.final_output
        except Exception as e:
            print(f"   ❌ Teacher Agent Error: {e}")
            break

        print(f"   🗣️ [Teacher]: {teacher_feedback[:100]}...")

        full_history.append({"role": "user", "content": teacher_feedback})

        # CHECK TERMINATION
        feedback_lower = teacher_feedback.lower()
        stop_signals = [ 
            "<end_conversation>"
        ]
        
        if any(sig in feedback_lower for sig in stop_signals):
            print("   ✅ Simulation ended by Teacher.")
            break
            
        next_student_input = teacher_feedback

    return full_history

def load_user_prompt_template(question, answer, tool_analysis, workflow):
    try:
        path = "user.md" if os.path.exists("user.md") else "../prompts/user.md"
        with open(path, "r") as f:
            template = f.read()
    except Exception:
        return ''

    if not isinstance(answer, str):
        answer = json.dumps(answer, indent=2)
    return template.replace("{QUESTION}", question).replace("{TOOL_OUTPUTS}", answer).replace("{TOOL_ANALYSIS}", tool_analysis).replace("{WORKFLOW_ANALYSIS}", workflow)

def setup_virtual_tools_for_item(item, client, tool_model_name):
    virtual_backend = VirtualToolBackend(client, tool_model_name)
    virtual_tools = []
    
    mcp_servers = item.get('metadata', {}).get('mcp_servers', [])
    seen_tools = set()

    for server in mcp_servers:
        tools_list = server.get('remote_server_response', {}).get('tools', []) or \
                     server.get('server_info_crawled', {}).get('tools', [])
            
        for tool_def in tools_list:
            t_name = tool_def.get('name')
            if t_name and t_name not in seen_tools:
                clean_def = sanitize_tool_schema(tool_def)
                v_tool = create_dynamic_virtual_tool(clean_def, virtual_backend)
                virtual_tools.append(v_tool)
                seen_tools.add(t_name)
                
    return virtual_tools, virtual_backend

async def generate_oracle_answer(client, args, question, item, tools):
    """Fallback: Oracle Generation"""
    print(f"   🔮 Oracle: Generating ground truth using {args.answer_gen_model}...")
    oracle_model = OpenAIResponsesModel(args.answer_gen_model, openai_client=client)
    
    metadata = item.get('metadata', {})
    target_tools = item.get("target_tools") or metadata.get("target_tools", "")
    if isinstance(target_tools, list): target_tools = ", ".join(target_tools)
    tool_analysis = item.get("tool_analysis") or metadata.get("tool_analysis", "")
    workflow = item.get("cross_tool_workflow") or item.get("cross_server_workflow") or metadata.get("cross_tool_workflow", "")

    instructions = (
        "You are an Expert Oracle. Your goal is to demonstrate the correct solution to the user's question.\n"
        "You MUST use the provided tools to verify information and generate the final response.\n"
        "Do NOT include internal reasoning tags in tool arguments.\n"
    )
    if target_tools: instructions += f"Target Tools: {target_tools}\n"
    if tool_analysis: instructions += f"Tool Selection Analysis: {tool_analysis}\n"
    if workflow: instructions += f"Recommended Workflow: {workflow}\n"
    instructions += "\nExecute the optimal tool calls now."
    
    oracle_agent = Agent(name="Oracle", instructions=instructions, model=oracle_model, tools=tools)
    result = await Runner.run(oracle_agent, input=question)
    
    print(f"   🔮 Oracle Answer: {result.final_output[:100]}...")
    return result.final_output

async def process_item(item, args, client):
    row_id = item.get('metadata', {}).get('row_id', 'unknown')
    print(f"🚀 Processing Row {row_id}")

    question = item.get("question", "")
    if not question:
        msgs = item.get("messages", [])
        question = next((m['content'] for m in msgs if m['role'] == 'user'), "")
    
    if not question:
        print("   ⚠️ Skipping: No question found.")
        return item

    virtual_tools, _ = setup_virtual_tools_for_item(item, client, args.virtual_tool_model)
    if not virtual_tools:
        print("   ⚠️ Skipping: No virtual tools found in metadata.")
        return item

    # --- DETERMINE GROUND TRUTH ---
    ground_truth_answer = ""
    
    if 'ground_truth_outputs' in item and item['ground_truth_outputs']:
        print(f"   ✅ Using pre-calculated deterministic tool outputs ({len(item['ground_truth_outputs'])} steps).")
        ground_truth_answer = format_ground_truth_for_teacher(item['ground_truth_outputs'])
    else:
        print("   ⚠️ No pre-calculated outputs found. Falling back to Oracle generation.")
        ground_truth_answer = await generate_oracle_answer(client, args, question, item, virtual_tools)
    metadata = item.get('metadata', {})
    tool_analysis = item.get("tool_analysis") or metadata.get("tool_analysis", "")
    workflow = item.get("cross_tool_workflow") or item.get("cross_server_workflow") or metadata.get("cross_tool_workflow", "")
    student_model = OpenAIResponsesModel(args.model_path, openai_client=client)
    student_agent = Agent(
        name="Student",
        instructions="You are a helpful assistant. Use the provided tools to answer the user query.",
        model=student_model,
        tools=virtual_tools
    )

    teacher_model = OpenAIResponsesModel(args.teacher_model, openai_client=client)
    teacher_instructions = load_user_prompt_template(question, ground_truth_answer, tool_analysis, workflow)
    
    teacher_agent = Agent(
        name="Teacher",
        instructions=teacher_instructions,
        model=teacher_model
    )

    full_trajectory = await run_student_teacher_loop(
        student_agent, 
        teacher_agent, 
        row_id, 
        initial_query=question,
        max_turns=args.max_turns
    )

    item['messages'] = full_trajectory
    if 'metadata' not in item: item['metadata'] = {}
    item['metadata']['ground_truth_used'] = ground_truth_answer
    item['metadata']['simulation_type'] = 'deterministic_student_teacher' if 'ground_truth_outputs' in item else 'oracle_student_teacher'
    
    return item

async def main():
    args = get_args()
    
    if not args.openrouter_api_key:
        print("❌ Error: OpenRouter API Key not found. Please set OPENROUTER_API_KEY env var.")
        return

    client = AsyncClient(base_url=args.openrouter_url, api_key=args.openrouter_api_key)
    
    print(f"📂 Loading Data from: {args.input_file}")
    dataset = load_dataset_from_file(args.input_file)
    if not isinstance(dataset, list): dataset = [dataset]

    output_file = args.input_file.replace(".jsonl", "_multiagent_results.jsonl")
    if output_file == args.input_file:
        output_file = args.input_file + "_results.jsonl"
    
    print(f"📝 Output will be streamed to: {output_file}")
    with open(output_file, 'w') as f:
        pass 

    for i, item in enumerate(dataset):
        try:
            result = await process_item(item, args, client)
            
            with open(output_file, 'a') as f:
                f.write(json.dumps(result) + "\n")
            
        except Exception as e:
            print(f"❌ Error processing item {i}: {e}")
            import traceback
            traceback.print_exc()

    print(f"✅ Simulation Complete. All rows saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())