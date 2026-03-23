import torch
import os
import sys
import argparse
import copy
import json
import re
import types
import asyncio
import base64
import signal
import atexit
import shutil
from time import time
from tqdm import tqdm
from virtual_tools import VirtualToolBackend, create_dynamic_virtual_tool

from utils import load_dataset_from_file, save_dataset, validate_api_pool_from_file, get_model_abbreviation


# Suppress OpenAI Agent SDK tracing logs before importing
import logging
logging.getLogger("openai.agents").setLevel(logging.ERROR)
logging.getLogger("agents").setLevel(logging.ERROR)
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# OpenAI Agent imports
from agents.mcp import MCPServerStreamableHttp
from agents.run_context import RunContextWrapper
from agents import Agent, OpenAIResponsesModel, Runner, SQLiteSession
from agents.tracing import set_tracing_disabled
set_tracing_disabled(True)
from openai import AsyncClient
from typing import Dict, Any, List, Optional

# Check if agents library is installed
try:
    import agents
except ImportError:
    print("agents library is not installed. Please install it.")
    exit(1)

# Global cleanup function for MCP resources
def cleanup_mcp_resources():
    try:
        if 'args' in globals() and hasattr(args, 'agent') and args.agent:
            pass
    except Exception:
        pass

def signal_handler(signum, frame):
    cleanup_mcp_resources()
    os._exit(0)

atexit.register(cleanup_mcp_resources)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

################
# Configurations
################
def get_args():
    parser = argparse.ArgumentParser(description="Multi-Agent Response Generation Manager.")
    parser.add_argument("--model_path", type=str, default="openai/gpt-4o",
                        help="Model path used for Student, User, and Virtual Tool simulation")
    parser.add_argument("--input_file", type=str, default=None, help="Input dataset file name")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index (inclusive) of rows to process.")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional number of rows to process from start_idx.")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="OpenAI-compatible API base URL ending with /v1.")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key for the endpoint.")
    parser.add_argument("--smithery_api_key", type=str, default="", help="Smithery API Key")
    parser.add_argument("--smithery_profile", type=str, default="", help="Smithery Profile")
    parser.add_argument("--smithery_api_pool", type=str, default="smithery_api_pool.json", help="Path to Smithery API pool JSON file")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of parallel workers")

    # Generation Parameters
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--step", type=str, default="unknown", help="Processing step identifier.")
    parser.add_argument("--agent", type=str, default="openai_agent", help="Use agent inference")
    parser.add_argument("--timeout", type=int, default=900, help="Timeout in seconds for each item processing")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries")
    parser.add_argument("--parallel_function_calls", type=bool, default=True, help="Parallel function calls")
    parser.add_argument("--reasoning_effort", type=str, default="high", help="Reasoning effort")
    parser.add_argument("--max_turns", type=int, default=15, help="Maximum agent tool-call turns per student response")

    # Tool parameters
    parser.add_argument("--virtual_tools", action="store_true", help="Use LLM-hallucinated tools instead of real MCP connections")
    parser.add_argument("--virtual_tool_model", type=str, default=None,
                    help="Model for virtual tool simulation (default: same as --model_path)")
    parser.add_argument("--mcp_server_dir", type=str, default="../mcp_servers/smithery_mcp_servers_0210",
                    help="Path to directory of MCP server JSON files")

    # Multi-agent specific parameters
    parser.add_argument("--user_model", type=str, default="openai/gpt-4o",
                    help="Deprecated: ignored; User agent uses --model_path")
    parser.add_argument("--user_max_turns", type=int, default=5,
                    help="Maximum number of Student-User conversation turns")
    parser.add_argument("--user_prompt_template", type=str, default=None,
                    help="Path to user prompt template (default: prompts/user.md relative to this script)")
    parser.add_argument("--student_prompt_template", type=str, default=None,
                    help="Path to student prompt template (default: prompts/student.md relative to this script)")

    return parser.parse_args()


def resolve_script_relative_path(path):
    if not path or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(SCRIPT_DIR, path))


args = get_args()
args.smithery_api_pool = resolve_script_relative_path(args.smithery_api_pool)
args.mcp_server_dir = resolve_script_relative_path(args.mcp_server_dir)
print(f"Multi-Agent Response Generation Manager. Arguments: {args}")

if args.input_file is None:
    raise ValueError("Please specify the input file path.")

if not args.input_file.endswith("prepared.jsonl") and not args.input_file.endswith("prepared.json"):
    print("Error: Input file must end with prepared.json(l). Please make sure you are using the correct input file.")
    exit(1)
if args.start_idx < 0:
    raise ValueError("--start_idx must be a non-negative integer.")
if args.batch_size is not None and args.batch_size <= 0:
    raise ValueError("--batch_size must be a positive integer.")

if args.user_model != args.model_path:
    print(
        f"⚠️  --user_model ({args.user_model}) is deprecated and ignored. "
        f"Using --model_path ({args.model_path}) for User agent."
    )
args.virtual_tool_model = args.virtual_tool_model or args.model_path

# Normalize base_url
normalized_base_url = args.base_url.rstrip("/").removesuffix("/chat/completions")
if not normalized_base_url.endswith("/v1"):
    raise ValueError("--base_url must end with /v1.")
args.base_url = normalized_base_url

# Constants
MODEL_NAME = args.model_path
INPUT_FILE_NAME = args.input_file

model_abbreviation = get_model_abbreviation(args.model_path)
config_str = f"{model_abbreviation}_multiagent_pfc" if args.parallel_function_calls else f"{model_abbreviation}_multiagent_sfc"

# Global API pool variable
smithery_api_pool = None

def load_and_validate_smithery_api_pool(pool_file_path):
    global smithery_api_pool
    print("=" * 50)
    print("🔍 SMITHERY API POOL CHECK (Non-blocking)")
    print("=" * 50)
    try:
        if not os.path.exists(pool_file_path):
            print(f"ℹ️  API pool file {pool_file_path} not found.")
            smithery_api_pool = []
            return []
        print(f"📁 Found {pool_file_path}. Attempting validation...")
        try:
            results = validate_api_pool_from_file(pool_file_path)
            if "error" in results:
                print(f"⚠️  API pool validation warning: {results['error']}")
                smithery_api_pool = []
                return []
            with open(pool_file_path, 'r') as f:
                original_data = json.load(f)
                original_pool = original_data.get('api_pool', [])
            valid_pool = []
            for result in results['results']:
                if result['valid']:
                    for original_entry in original_pool:
                        if original_entry['profile'] == result['profile']:
                            valid_pool.append(original_entry)
                            break
            smithery_api_pool = valid_pool
            print(f"✅ Loaded {len(smithery_api_pool)} valid API keys from pool.")
            return smithery_api_pool
        except Exception as e:
            print(f"⚠️  Validation check failed: {e}")
            smithery_api_pool = []
            return []
    except Exception as e:
        print(f"⚠️  Unexpected error loading pool: {e}")
        smithery_api_pool = []
        return []

def get_api_key_for_worker(worker_id):
    if smithery_api_pool and len(smithery_api_pool) > 0:
        pool_entry = smithery_api_pool[worker_id % len(smithery_api_pool)]
        return pool_entry['api_key'], pool_entry['profile']
    else:
        return args.smithery_api_key, args.smithery_profile

def construct_mcp_server_url(server_info, api_key=None, profile=None):
    if not server_info:
        return None
    server_url = server_info.get('python_sdk_url', '')
    if not server_url:
        return None
    if api_key is None:
        api_key = args.smithery_api_key
    if profile is None:
        profile = args.smithery_profile
    mcp_config = server_info.get('python_sdk_config', "")
    if mcp_config == "":
        mcp_config = {"debug": False}
    else:
        try:
            mcp_config = json.loads(mcp_config)
        except json.JSONDecodeError:
            mcp_config = {"debug": False}
    config_b64 = base64.b64encode(json.dumps(mcp_config).encode()).decode()
    if "{config_b64}" in server_url:
        server_url = server_url.replace("{config_b64}", config_b64)
    if "{smithery_api_key}" in server_url:
        server_url = server_url.replace("{smithery_api_key}", api_key)
    if "{smithery_profile}" in server_url:
        server_url = server_url.replace("{smithery_profile}", profile)
    elif "&profile=" not in server_url and "profile=" not in server_url:
        server_url += f"&profile={profile}"
    return server_url

def construct_mcp_url_from_source(server_info, api_key=None, profile=None):
    if api_key is None:
        api_key = args.smithery_api_key
    if profile is None:
        profile = args.smithery_profile
    deployment_url = None
    source_path = server_info.get('source_file_path', '')
    if source_path:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            resolved_path = os.path.normpath(os.path.join(script_dir, source_path))
            if os.path.exists(resolved_path):
                with open(resolved_path, 'r') as f:
                    source_data = json.load(f)
                server_data = source_data.get('server', {})
                connections = server_data.get('connections', [])
                if connections:
                    deployment_url = connections[0].get('deploymentUrl', '')
                if not deployment_url:
                    dep = server_data.get('deploymentUrl', '')
                    if dep:
                        deployment_url = dep.rstrip('/') + '/mcp'
        except Exception as e:
            print(f"⚠️ Warning: Could not load source file {source_path}: {e}")
    if not deployment_url:
        return None
    config = {"debug": False}
    config_b64 = base64.b64encode(json.dumps(config).encode()).decode()
    server_url = f"{deployment_url}?config={config_b64}&api_key={api_key}&profile={profile}"
    return server_url


################
# User Prompt Utilities
################

def load_user_prompt_template():
    """Load the user.md prompt template."""
    if args.user_prompt_template:
        path = args.user_prompt_template
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, 'prompts', 'user.md')
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# Load user template once at module level
USER_PROMPT_TEMPLATE = load_user_prompt_template()


def load_student_prompt_template():
    """Load the student.md prompt template."""
    if args.student_prompt_template:
        path = args.student_prompt_template
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, 'prompts', 'student.md')
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# Load student template once at module level
STUDENT_PROMPT_TEMPLATE = load_student_prompt_template()


def format_target_tool_outputs(target_tools):
    """Format target_tools list into readable ground truth for the user agent."""
    if not target_tools or not isinstance(target_tools, list):
        return "No ground truth tool outputs available."
    parts = []
    for i, tool_entry in enumerate(target_tools, 1):
        if isinstance(tool_entry, dict):
            server = tool_entry.get('server', '')
            tool = tool_entry.get('tool', '')
            output = tool_entry.get('output', '')
            parts.append(f"Tool {i}: {server}::{tool}\nOutput: {output}")
    return "\n\n".join(parts) if parts else "No ground truth tool outputs available."


def format_withheld_info(withheld_info):
    """Format withheld_info list into a section for the user prompt template."""
    if not withheld_info or not isinstance(withheld_info, list):
        return ""
    lines = []
    for item in withheld_info:
        if isinstance(item, dict):
            param = item.get('parameter', '')
            desc = item.get('description', '')
            value = item.get('value', '')
            lines.append(f'- **{param}** ({desc}): When asked, answer: "{value}"')
    if not lines:
        return ""
    section = (
        "\n<withheld_information>\n"
        "The following information was NOT included in the initial query. "
        "Provide it naturally as the user when the assistant asks for it:\n"
        + "\n".join(lines)
        + "\n</withheld_information>\n"
    )
    return section


def format_tool_descriptions(metadata):
    """Build a concise tool-description block from mcp_servers metadata."""
    mcp_servers = metadata.get('mcp_servers', [])
    lines = []
    for srv in mcp_servers:
        srv_name = srv.get('server_name', 'Unknown')
        for tool in srv.get('tools', []):
            name = tool.get('name', '?')
            desc = tool.get('description', '') or '(no description)'
            lines.append(f"- **{srv_name}::{name}**: {desc}")
    return "\n".join(lines) if lines else "No tool descriptions available."


def build_user_agent_instructions(metadata):
    """Build user agent instructions from item metadata."""
    question = metadata.get('question', '')
    target_tools = metadata.get('target_tools', [])
    tool_analysis = metadata.get('tool_analysis', '')
    workflow = metadata.get('cross_tool_workflow', '')
    withheld_info = metadata.get('withheld_info', [])

    withheld_str = format_withheld_info(withheld_info)
    tool_desc_str = format_tool_descriptions(metadata)

    instructions = (
        USER_PROMPT_TEMPLATE
        .replace("{QUESTION}", question)
        .replace("{TOOL_ANALYSIS}", tool_analysis)
        .replace("{WORKFLOW_ANALYSIS}", workflow)
        .replace("{WITHHELD_INFO}", withheld_str)
        .replace("{TOOL_DESCRIPTIONS}", tool_desc_str)
    )
    return instructions


################
# Agent Creation
################

def create_student_agent_config(item, client, api_key=None, profile=None):
    """
    Create configuration for the Student (tool-using) agent.
    Returns a dict with 'model', 'tools' (virtual) or 'mcp_servers_list' (real).
    """
    metadata = item.get('metadata', {})
    mcp_servers = metadata.get('mcp_servers', [])

    if not mcp_servers or not isinstance(mcp_servers, list):
        return None

    model = OpenAIResponsesModel(args.model_path, openai_client=client)

    # --- VIRTUAL TOOLS ---
    if args.virtual_tools:
        print(f"👻 Configuring Student with VIRTUAL tools (Agent/User: {args.model_path}, VirtualTool: {args.virtual_tool_model})...")
        virtual_backend = VirtualToolBackend(client, model_path=args.virtual_tool_model)
        virtual_tool_funcs = []
        registered_tool_names = set()
        tool_simulation_messages = []  # shared across all tools for cross-tool continuity

        # Build scenario context for virtual tools from metadata
        question = metadata.get('question', '')
        tool_analysis = metadata.get('tool_analysis', '')
        workflow_analysis = metadata.get('cross_tool_workflow', '')
        target_tools = metadata.get('target_tools', [])
        expected_outputs_by_tool = {}
        required_tool_names = set()
        for tt in (target_tools or []):
            if isinstance(tt, dict):
                tool_key = tt.get('tool', '')
                if tool_key:
                    normalized_tool_key = tool_key.split("::", 1)[-1]
                    expected_outputs_by_tool[normalized_tool_key] = tt.get('output', '')
                    required_tool_names.add(normalized_tool_key)

        virtual_tool_diagnostics = {
            "total_tools_discovered": 0,
            "tools_registered": 0,
            "required_tools_expected": sorted(required_tool_names),
            "required_tool_errors": [],
            "optional_tool_errors": [],
        }

        for server_info in mcp_servers:
            server_id = server_info.get('server_id', '')
            server_name = server_info.get('server_name', '')
            server_analysis = server_info.get('server_description', '')
            tools_list = server_info.get('tools', [])

            if args.mcp_server_dir and server_id:
                smithery_path = os.path.join(args.mcp_server_dir, f"{server_id}.json")
                if os.path.exists(smithery_path):
                    with open(smithery_path) as sf:
                        smithery_data = json.load(sf)
                    smithery_tools = smithery_data.get('server', {}).get('tools', [])
                    tools_list = smithery_tools if smithery_tools else tools_list
                    server_analysis = smithery_data.get('analysis', server_analysis)
                    server_name = smithery_data.get('server', {}).get('displayName', server_name)

            for tool_def in tools_list:
                virtual_tool_diagnostics["total_tools_discovered"] += 1
                if server_analysis and server_name and tool_def.get('description'):
                    tool_def = dict(tool_def)
                    tool_def['description'] = (
                        f"This tool comes from the MCP server: {server_name}.\n\n"
                        f"An analysis of this server is as follows: {server_analysis}.\n\n"
                        f"This tool has the following functionality within the MCP server: {tool_def['description']}"
                    )
                # Build scenario context for this tool
                tool_raw_name = tool_def.get('name', '')
                expected_output = expected_outputs_by_tool.get(tool_raw_name, '')
                scenario_ctx = {
                    'question': question,
                    'tool_analysis': tool_analysis,
                    'workflow_analysis': workflow_analysis,
                    'expected_output': expected_output,
                    'tool_simulation_messages': tool_simulation_messages,
                }
                is_required = tool_raw_name in required_tool_names
                try:
                    v_tool = create_dynamic_virtual_tool(
                        tool_def,
                        virtual_backend,
                        scenario_context=scenario_ctx,
                    )
                    virtual_tool_funcs.append(v_tool)
                    registered_tool_names.add(tool_raw_name)
                except Exception as tool_error:
                    err_entry = {
                        "tool": tool_raw_name,
                        "server_id": server_id,
                        "server_name": server_name,
                        "error": str(tool_error),
                    }
                    if is_required:
                        virtual_tool_diagnostics["required_tool_errors"].append(err_entry)
                    else:
                        virtual_tool_diagnostics["optional_tool_errors"].append(err_entry)
                    print(
                        f"⚠️ Skipping virtual tool '{tool_raw_name}' "
                        f"(required={is_required}) due to registration error: {tool_error}"
                    )

        missing_required_tools = sorted(required_tool_names - registered_tool_names)
        virtual_tool_diagnostics["tools_registered"] = len(virtual_tool_funcs)
        virtual_tool_diagnostics["missing_required_tools"] = missing_required_tools
        virtual_tool_diagnostics["tools_skipped_required"] = len(virtual_tool_diagnostics["required_tool_errors"])
        virtual_tool_diagnostics["tools_skipped_optional"] = len(virtual_tool_diagnostics["optional_tool_errors"])
        metadata["virtual_tool_diagnostics"] = virtual_tool_diagnostics

        if missing_required_tools:
            raise ValueError(
                "Required virtual tool(s) failed to register: "
                + ", ".join(missing_required_tools)
            )

        if not virtual_tool_funcs:
            print("❌ No tool definitions found for virtual generation.")
            return None

        return {
            "name": "Student-Virtual-Assistant",
            "instructions": STUDENT_PROMPT_TEMPLATE,
            "model": model,
            "tools": virtual_tool_funcs,
            "mcp_servers_list": [],
            "virtual_tool_diagnostics": virtual_tool_diagnostics,
        }

    # --- REAL MCP SERVERS ---
    else:
        mcp_servers_list = []
        for server_info in mcp_servers:
            server_url = None
            server_details = server_info.get('server_info', {})
            if server_details:
                server_url = construct_mcp_server_url(server_details, api_key, profile)
            if not server_url:
                server_url = construct_mcp_url_from_source(server_info, api_key, profile)
            if server_url:
                safe_name = server_info.get('server_name', 'unknown').replace(' ', '-').lower()
                mcp_servers_list.append({
                    "name": safe_name,
                    "url": server_url,
                    "timeout": 600.0,
                    "sse_read_timeout": 600.0,
                    "terminate_on_close": False
                })

        if not mcp_servers_list:
            return None

        return {
            "name": "Student-Assistant",
            "instructions": STUDENT_PROMPT_TEMPLATE,
            "model": model,
            "mcp_servers_list": mcp_servers_list
        }


def qwen_compatible_system_prompt_generator(tools):
    """Generate a Qwen-compatible system prompt from tool specs."""
    function_schemas = []
    for tool in tools or []:
        name = getattr(tool, 'name', None) or ''
        description = getattr(tool, 'description', None) or ''
        params_schema = getattr(tool, 'params_json_schema', None) or {"type": "object", "properties": {}}
        function_schemas.append({"name": name, "description": description, "parameters": params_schema})

    tool_descs_wrapped = [{"type": "function", "function": fs} for fs in function_schemas]
    tool_descs_str = "\n".join(json.dumps(d, ensure_ascii=False) for d in tool_descs_wrapped)

    return (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{tool_descs_str}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
        "</tool_call>"
    )


################
# Message Extraction
################

def extract_new_messages_from_result(result):
    """
    Extract only the NEW messages (tool calls, outputs, assistant text) from an agent result.
    Returns a flat list of message dicts without duplicating original conversation history.
    """
    new_messages = []

    if not hasattr(result, 'new_items') or not result.new_items:
        if result.final_output:
            new_messages.append({"role": "assistant", "content": result.final_output})
        return new_messages

    current_reasoning = []
    matched_call_ids = set()

    for item_flow in result.new_items:
        if item_flow.type == "reasoning_item":
            raw_content = getattr(getattr(item_flow, 'raw_item', None), 'content', None)
            if isinstance(raw_content, list):
                for content in raw_content:
                    if hasattr(content, 'text'):
                        current_reasoning.append(content.text)

        elif item_flow.type == "tool_call_item":
            if hasattr(item_flow, 'raw_item'):
                tool_call = {
                    "name": getattr(item_flow.raw_item, 'name', None),
                    "arguments": getattr(item_flow.raw_item, 'arguments', None),
                    "call_id": getattr(item_flow.raw_item, 'call_id', None)
                }
                if current_reasoning:
                    new_messages.append({
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "\n".join(current_reasoning)
                    })
                    current_reasoning = []
                new_messages.append({
                    "role": "assistant",
                    "content": "",
                    "function_call": tool_call
                })

        elif item_flow.type == "tool_call_output_item":
            if hasattr(item_flow, 'output'):
                try:
                    output_data = json.loads(item_flow.output)
                    if output_data.get('type') == 'text':
                        inner_data = json.loads(output_data.get('text', '{}'))
                        tool_output = json.dumps(inner_data)
                    else:
                        tool_output = item_flow.output
                except Exception:
                    tool_output = item_flow.output

                tool_name = 'unknown'
                matched_call_id = None
                if hasattr(item_flow, 'raw_item'):
                    raw = item_flow.raw_item
                    call_id = None
                    for attr in ['tool_call_id', 'call_id', 'id', 'toolCallId']:
                        if hasattr(raw, attr):
                            call_id = getattr(raw, attr)
                            break
                    if call_id is not None:
                        for prev_msg in reversed(new_messages):
                            if (prev_msg.get('role') == 'assistant' and
                                    'function_call' in prev_msg and
                                    prev_msg['function_call'].get('call_id') == call_id):
                                tool_name = prev_msg['function_call'].get('name', 'unknown')
                                matched_call_id = call_id
                                break

                if tool_name == 'unknown':
                    for prev_msg in new_messages:
                        if prev_msg.get('role') == 'assistant' and 'function_call' in prev_msg:
                            fc = prev_msg['function_call']
                            fc_call_id = fc.get('call_id')
                            if fc_call_id not in matched_call_ids:
                                name_candidate = fc.get('name')
                                if name_candidate:
                                    tool_name = name_candidate
                                    matched_call_id = fc_call_id
                                    break

                if matched_call_id is not None:
                    matched_call_ids.add(matched_call_id)

                new_messages.append({
                    "role": "function",
                    "content": tool_output,
                    "name": tool_name
                })

        elif item_flow.type == "message_output_item":
            raw_content = getattr(getattr(item_flow, 'raw_item', None), 'content', None)
            if isinstance(raw_content, list):
                message_texts = []
                for content in raw_content:
                    if hasattr(content, 'text'):
                        message_texts.append(content.text)

                if current_reasoning:
                    new_messages.append({
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "\n".join(current_reasoning)
                    })
                    current_reasoning = []

                final_content = "\n".join(message_texts)
                if final_content.strip():
                    new_messages.append({
                        "role": "assistant",
                        "content": final_content
                    })

    # Flush remaining reasoning
    if current_reasoning:
        new_messages.append({
            "role": "assistant",
            "content": "",
            "reasoning_content": "\n".join(current_reasoning)
        })

    # Fallback: if no substantive assistant message was produced, use final_output
    if not any(msg.get('role') == 'assistant' and msg.get('content') for msg in new_messages):
        if result.final_output:
            new_messages.append({"role": "assistant", "content": result.final_output})

    return new_messages


################
# Multi-Agent Loop
################

async def run_student_user_loop_async(
    student_agent_config,
    user_instructions,
    question,
    user_client,
    session_prefix,
    mcp_server_contexts=None,
):
    """
    Run the alternating Student–User multi-turn loop.

    Returns (conversation_messages, system_prompt) where:
    - conversation_messages: full trajectory as list of message dicts
    - system_prompt: generated from student's available tools
    """
    # --- Create Student agent ---
    agent_kwargs = {
        "name": student_agent_config["name"],
        "instructions": student_agent_config["instructions"],
        "model": student_agent_config["model"],
    }
    if student_agent_config.get("tools"):
        agent_kwargs["tools"] = student_agent_config["tools"]
    if mcp_server_contexts:
        agent_kwargs["mcp_servers"] = mcp_server_contexts

    student_agent = Agent(**agent_kwargs)
    run_context = RunContextWrapper(context=None)

    # --- Create User agent (no tools) ---
    user_model = OpenAIResponsesModel(args.model_path, openai_client=user_client)
    user_agent = Agent(
        name="User",
        instructions=user_instructions,
        model=user_model,
    )

    # --- Initialize sessions ---
    student_session = SQLiteSession(f"student_{session_prefix}")
    await student_session.clear_session()

    user_session = SQLiteSession(f"user_{session_prefix}")
    await user_session.clear_session()

    # Pre-fill User session: User "already said" the initial question
    await user_session.add_items([{"role": "assistant", "content": question}])

    # --- Build conversation trajectory ---
    conversation_messages = [{"role": "user", "content": question}]
    system_prompt = None
    user_feedback = None

    for turn in range(args.user_max_turns):
        print(f"    Turn {turn + 1}/{args.user_max_turns}")

        # --- STUDENT TURN ---
        student_input = question if turn == 0 else user_feedback

        student_result = await Runner.run(
            student_agent,
            input=student_input,
            session=student_session,
            max_turns=args.max_turns,
        )

        # Generate system prompt once from first turn
        if system_prompt is None:
            available_tools = await student_agent.get_all_tools(run_context)
            system_prompt = qwen_compatible_system_prompt_generator(available_tools)

        # Extract new messages from this student turn
        new_msgs = extract_new_messages_from_result(student_result)
        conversation_messages.extend(new_msgs)

        student_reply = student_result.final_output or ""
        print(f"    [Student]: {student_reply[:120]}...")

        # --- USER TURN ---
        user_input = f"The assistant responded: {student_reply}"

        user_result = await Runner.run(
            user_agent,
            input=user_input,
            session=user_session,
        )
        user_feedback = user_result.final_output or ""
        print(f"    [User]: {user_feedback[:120]}...")

        # Add user message to conversation trajectory
        conversation_messages.append({"role": "user", "content": user_feedback})

        # Check for termination signal
        if "<end_conversation>" in user_feedback.lower():
            print("    Conversation ended by User agent.")
            break

    # Prepend system prompt
    if system_prompt:
        conversation_messages = [{"role": "system", "content": system_prompt}] + conversation_messages

    return conversation_messages


################
# Single Item Processing
################

async def process_single_item_multiagent_async(item, client, api_key=None, profile=None):
    """Process a single item using the multi-agent Student–User loop."""
    metadata = item.get('metadata', {})
    prompt_id = metadata.get('prompt_id', 'unknown')

    # Extract question
    question = metadata.get('question', '')
    if not question:
        messages = item.get('messages', [])
        user_msgs = [m for m in messages if m.get('role') == 'user']
        question = user_msgs[-1]['content'] if user_msgs else ''
    if not question:
        raise ValueError(f"No question found for item {prompt_id}")

    # Build user agent instructions
    user_instructions = build_user_agent_instructions(metadata)

    # Create student agent configuration
    student_config = create_student_agent_config(item, client, api_key, profile)
    if student_config is None:
        raise ValueError(f"Could not create student agent config for item {prompt_id}")
    if student_config.get("virtual_tool_diagnostics"):
        metadata["virtual_tool_diagnostics"] = student_config["virtual_tool_diagnostics"]

    print(f"🚀 Running multi-agent inference for item {prompt_id}...")

    mcp_server_contexts = []
    server_context_managers = []

    try:
        # Set up MCP servers if using real tools
        server_configs = student_config.get("mcp_servers_list", [])
        if server_configs:
            for server_config in server_configs:
                try:
                    mcp_server_ctx = MCPServerStreamableHttp(
                        name=server_config["name"],
                        params={
                            "url": server_config["url"],
                            "headers": {
                                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
                            },
                            "timeout": server_config.get("timeout", 600.0),
                            "sse_read_timeout": server_config.get("sse_read_timeout", 600.0),
                            "terminate_on_close": server_config.get("terminate_on_close", False)
                        },
                        client_session_timeout_seconds=args.timeout,
                    )
                    mcp_server = await mcp_server_ctx.__aenter__()
                    mcp_server_contexts.append(mcp_server)
                    server_context_managers.append(mcp_server_ctx)
                except Exception as conn_error:
                    print(f"⚠️ Skipping MCP server '{server_config['name']}': {conn_error}")

            if not mcp_server_contexts and not student_config.get("tools"):
                raise Exception(f"All {len(server_configs)} MCP server(s) failed to connect")

        # Run the Student–User loop
        conversation_messages = await run_student_user_loop_async(
            student_agent_config=student_config,
            user_instructions=user_instructions,
            question=question,
            user_client=client,
            session_prefix=str(prompt_id),
            mcp_server_contexts=mcp_server_contexts if mcp_server_contexts else None,
        )

        if len(conversation_messages) > 1:
            print(f"✅ Multi-agent inference completed for item {prompt_id} ({len(conversation_messages)} messages)")
            item['messages'] = conversation_messages
        else:
            raise Exception("Multi-agent loop returned empty conversation")

    finally:
        for ctx in reversed(server_context_managers):
            try:
                await ctx.__aexit__(None, None, None)
            except Exception as cleanup_error:
                print(f"⚠️ Warning: Failed to cleanup MCP server: {cleanup_error}")

    return item


def build_error_item(item, error_text):
    fallback_item = copy.deepcopy(item)
    original_messages = fallback_item.get("messages", [])
    fallback_item["messages"] = original_messages + [
        {"role": "assistant", "content": f"[ERROR: {error_text}]"}
    ]
    return fallback_item


def sort_dataset_by_row_id(dataset):
    def get_sort_key(item):
        metadata = item.get('metadata', {})
        row_id = metadata.get('row_id')
        if row_id is not None:
            try:
                return int(row_id)
            except (ValueError, TypeError):
                return float('inf'), str(row_id)
        return float('inf'), ''
    return sorted(dataset, key=get_sort_key)


def get_input_base_name(input_file):
    base_name = input_file[: input_file.rfind('.')]
    if base_name.endswith("_4prepared"):
        return base_name[:-10]
    if base_name.endswith("_prepared"):
        return base_name[:-9]
    return base_name


def resolve_processing_range(total_rows):
    if total_rows <= 0:
        raise ValueError(f"Input dataset is empty: {args.input_file}")
    if args.start_idx >= total_rows:
        raise ValueError(
            f"--start_idx ({args.start_idx}) must be smaller than dataset size ({total_rows})."
        )

    start_idx = args.start_idx
    requested_end_idx = total_rows if args.batch_size is None else args.start_idx + args.batch_size
    end_idx = min(requested_end_idx, total_rows)
    return start_idx, requested_end_idx, end_idx


def build_output_paths(base_name, start_idx, end_idx, trial_idx=None):
    trial_suffix = f"{trial_idx}" if trial_idx is not None else ""
    range_mode = args.batch_size is not None or args.start_idx != 0
    if range_mode:
        saved_file = f"{base_name}_{config_str}_results{trial_suffix}_{start_idx}_{end_idx}.jsonl"
        checkpoint_dir = f"{base_name}_{config_str}_results{trial_suffix}_checkpoints_{start_idx}_{end_idx}"
    else:
        saved_file = f"{base_name}_{config_str}_results{trial_suffix}.jsonl"
        checkpoint_dir = f"{base_name}_{config_str}_results{trial_suffix}_checkpoints"
    return saved_file, checkpoint_dir


def add_generation_config_to_metadata(dataset, model_short_name, generation_params):
    config_entry = {
        "model": model_short_name,
        "generation_params": generation_params,
        "timestamp": int(time())
    }
    for item in dataset:
        if "metadata" not in item:
            item["metadata"] = {}
        if "synthetic_data_gen_configs" not in item["metadata"]:
            item["metadata"]["synthetic_data_gen_configs"] = []
        item["metadata"]["synthetic_data_gen_configs"].append(config_entry)
    return dataset


def checkpoint_file_path(index, checkpoint_dir):
    return os.path.join(checkpoint_dir, f"{index:08d}.json")


def save_item_checkpoint(index, item, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = checkpoint_file_path(index, checkpoint_dir)
    temp_path = f"{checkpoint_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")
    os.replace(temp_path, checkpoint_path)


def load_item_checkpoints(processed_dataset, checkpoint_dir):
    completed_indices = set()
    if not os.path.isdir(checkpoint_dir):
        return completed_indices

    for file_name in os.listdir(checkpoint_dir):
        match = re.fullmatch(r"(\d+)\.json", file_name)
        if match is None:
            continue

        index = int(match.group(1))
        if index < 0 or index >= len(processed_dataset):
            continue

        file_path = os.path.join(checkpoint_dir, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                processed_dataset[index] = json.load(f)
            completed_indices.add(index)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Failed to load checkpoint {file_path}: {e}. Reprocessing index {index}.")

    return completed_indices


def build_generation_params(max_workers):
    return {
        "base_url": args.base_url,
        "model_path": args.model_path,
        "student_model": args.model_path,
        "user_model": args.model_path,
        "virtual_tool_model": args.virtual_tool_model,
        "user_max_turns": args.user_max_turns,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "num_trials": args.num_trials,
        "step": args.step,
        "timeout": args.timeout,
        "max_workers": max_workers,
        "virtual_tools": args.virtual_tools,
        "start_idx": args.start_idx,
        "batch_size": args.batch_size,
    }


async def process_index_async(index, processed_dataset, client, semaphore):
    async with semaphore:
        api_key, profile = get_api_key_for_worker(index)
        current_item = copy.deepcopy(processed_dataset[index])
        prompt_id = current_item.get("metadata", {}).get("prompt_id", f"item_{index}")

        try:
            processed_item = await asyncio.wait_for(
                process_single_item_multiagent_async(current_item, client, api_key, profile),
                timeout=args.timeout,
            )
            print(f"✅ Completed item {prompt_id} (index {index})")
            return index, processed_item
        except Exception as e:
            if isinstance(e, asyncio.TimeoutError):
                error_text = f"Timed out after {args.timeout}s"
            else:
                error_text = str(e)
            print(f"❌ Failed item {prompt_id} (index {index}): {error_text}")
            return index, build_error_item(current_item, error_text)


async def generate_and_update(dataset, client, checkpoint_dir):
    processed_dataset = copy.deepcopy(dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)

    completed_indices = load_item_checkpoints(processed_dataset, checkpoint_dir)
    if completed_indices:
        print(
            f"Loaded {len(completed_indices)} completed item checkpoints from {checkpoint_dir}."
        )

    pending_indices = [idx for idx in range(len(processed_dataset)) if idx not in completed_indices]
    max_workers = args.max_workers or (len(smithery_api_pool) if smithery_api_pool else 8)
    print(f"Total items in dataset: {len(processed_dataset)}")
    print(f"Already completed: {len(completed_indices)}")
    print(f"Remaining to process: {len(pending_indices)}")

    if not pending_indices:
        print("No remaining items to process.")
    else:
        print(f"🚀 Starting multi-agent processing with {max_workers} workers...")
        print("💾 Checkpointing: One JSON file per completed item")
        print(f"⏱️ Item timeout: {args.timeout}s | Max user-agent turns: {args.user_max_turns}")

        start_time = time()
        semaphore = asyncio.Semaphore(max_workers)
        tasks = [
            asyncio.create_task(process_index_async(idx, processed_dataset, client, semaphore))
            for idx in pending_indices
        ]

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating completions"):
            index, processed_item = await task
            processed_dataset[index] = processed_item
            save_item_checkpoint(index, processed_item, checkpoint_dir)

        end_time = time()
        print(f"\n🎉 Multi-agent processing completed!")
        print(f"📊 Items processed: {len(pending_indices)}/{len(pending_indices)}")
        print(f"⏱️ Total time: {end_time - start_time:.2f} seconds")

    generation_params = build_generation_params(max_workers)
    processed_dataset = add_generation_config_to_metadata(processed_dataset, model_abbreviation, generation_params)
    return sort_dataset_by_row_id(processed_dataset)


async def run_trial(target_dataset, saved_file, checkpoint_dir):
    client = AsyncClient(base_url=args.base_url, api_key=args.api_key)
    try:
        updated_dataset = await generate_and_update(target_dataset, client, checkpoint_dir)
        save_dataset(updated_dataset, saved_file, convert_to_jsonl=True)
        if os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        print(f"Final dataset saved to {saved_file}.")
    finally:
        await client.close()


async def main():
    if args.num_trials <= 0:
        raise ValueError("--num_trials must be a positive integer.")

    api_pool = load_and_validate_smithery_api_pool(args.smithery_api_pool)

    pool_size = len(api_pool) if api_pool else 0
    effective_workers = args.max_workers or (pool_size if pool_size > 0 else 8)
    print("=" * 50)
    print("🚀 MULTI-AGENT PROCESSING CONFIGURATION")
    print("=" * 50)
    print(f"Model (Student/User/VirtualTool): {args.model_path}")
    print(f"User max turns: {args.user_max_turns}")
    print(f"Max agent turns per student response: {args.max_turns}")
    print(f"Tool mode:     {'Virtual' if args.virtual_tools else 'Real MCP'}")
    print(f"Endpoint: {args.base_url}")
    print(f"Workers: {effective_workers}")
    print(f"Timeout per item: {args.timeout} seconds")
    print("Checkpointing: One JSON file per completed item")
    print("=" * 50)

    dataset = load_dataset_from_file(INPUT_FILE_NAME)
    if not isinstance(dataset, list):
        dataset = [dataset]

    total_rows = len(dataset)
    start_idx, requested_end_idx, end_idx = resolve_processing_range(total_rows)
    target_dataset = dataset[start_idx:end_idx]
    base_name = get_input_base_name(args.input_file)

    print(
        f"Dataset rows: {total_rows}. Requested range: "
        f"[{start_idx}, {requested_end_idx}). Effective range: [{start_idx}, {end_idx})."
    )
    print(f"Processing {len(target_dataset)} rows.")

    if args.num_trials == 1:
        saved_file, checkpoint_dir = build_output_paths(base_name, start_idx, end_idx)
        print(f"Output file: {saved_file}")
        print(f"Checkpoint dir: {checkpoint_dir}")
        await run_trial(target_dataset, saved_file, checkpoint_dir)
    else:
        for trial_idx in range(args.num_trials):
            saved_file, checkpoint_dir = build_output_paths(base_name, start_idx, end_idx, trial_idx=trial_idx)
            print(f"Trial {trial_idx}: output={saved_file}")
            print(f"Trial {trial_idx}: checkpoints={checkpoint_dir}")
            await run_trial(target_dataset, saved_file, checkpoint_dir)

    print("🎯 Program execution completed.")


if __name__ == "__main__":
    asyncio.run(main())
