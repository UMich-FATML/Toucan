import json
import asyncio
from typing import Dict, Any, List, Callable, Optional
from openai import AsyncClient

from agents import FunctionTool, RunContextWrapper

def load_prompt_template(template_path):
  """Load the prompt template from file."""
  with open(template_path, 'r', encoding='utf-8') as f:
    return f.read()

class VirtualToolBackend:
    """
    Manages the LLM calls for generating virtual tool responses.
    """
    def __init__(self, client: AsyncClient, model_path: str):
        self.client = client
        self.model = model_path

    async def generate_response(self, tool_name: str, tool_doc: Dict, tool_args: Dict,
                                scenario_context: Optional[Dict] = None) -> Dict:
        """
        Hits the LLM to hallucinate a response for the tool.
        If scenario_context is provided, includes the user question and
        a reference expected output to guide realistic generation.
        """
        user_prompt = (
            f"Tool Documentation: {json.dumps(tool_doc)}\n\n"
            f"Generate a realistic JSON response for the following input:\n"
            f"{json.dumps(tool_args)}"
        )

        if scenario_context:
            question = scenario_context.get('question', '')
            tool_analysis = scenario_context.get('tool_analysis', '')
            workflow_analysis = scenario_context.get('workflow_analysis', '')
            expected_output = scenario_context.get('expected_output', '')
            if question:
                user_prompt += f"\n\nScenario context (the user's original request): {question}"
            if tool_analysis:
                user_prompt += f"\n\nTool analysis (how the tools relate to this scenario): {tool_analysis}"
            if workflow_analysis:
                user_prompt += f"\n\nWorkflow analysis (how tools chain together): {workflow_analysis}"
            if expected_output:
                user_prompt += (
                    f"\n\nGeneric reference output (use as a loose guide for style and "
                    f"detail level, but produce your response holistically based on "
                    f"the full scenario context above — this reference is NOT a hard "
                    f"requirement):\n{expected_output}"
                )

        # Ensure prompt file exists or use default
        system_content = load_prompt_template('./prompts/virtual_toucan.md')

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content

            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = {"result": content, "error": "Failed to parse JSON output"}

            return result

        except Exception as e:
            print(f"⚠️ Virtual Tool Generation Failed for {tool_name}: {e}")
            return {"error": str(e), "status": "failed"}


def _make_strict_schema(input_schema: Dict) -> Dict:
    """
    Build a strict-mode-compatible JSON schema from a smithery inputSchema.
    Ensures additionalProperties: false at every object level and that
    required/properties are present.
    """
    schema = dict(input_schema)
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema.setdefault("required", [])
    schema["additionalProperties"] = False

    # Recursively fix nested object properties
    for prop_name, prop_def in schema["properties"].items():
        if isinstance(prop_def, dict) and prop_def.get("type") == "object":
            schema["properties"][prop_name] = _make_strict_schema(prop_def)

    return schema


def create_dynamic_virtual_tool(tool_def: Dict, backend: VirtualToolBackend,
                               scenario_context: Optional[Dict] = None):
    """
    Dynamically creates a FunctionTool compatible with openai-agents.
    Uses the raw smithery inputSchema as params_json_schema so the LLM
    sees accurate type/items/required constraints (no lossy Pydantic conversion).

    Tool name sanitization: dots are replaced with underscores because the OpenAI
    API only allows [a-zA-Z0-9_-] in function names. The original name is preserved
    in tool_def for the virtual backend so responses stay contextually accurate.

    Strict schema fallback: if FunctionTool rejects the schema in strict mode
    (e.g. schemas with additionalProperties or unsupported nested types), we retry
    with strict_json_schema=False so the tool is still registered rather than
    crashing the entire item.
    """
    raw_name = tool_def.get('name', '')
    # Sanitize: replace dots (and any other chars invalid for OpenAI function names)
    tool_name = raw_name.replace('.', '_')
    tool_desc = tool_def.get('description', '')
    input_schema = tool_def.get('inputSchema', tool_def.get('input_schema', {}))

    params_schema = _make_strict_schema(input_schema)

    async def dynamic_run_function(ctx: RunContextWrapper[Any], args: str) -> str:
        """
        Executes the virtual tool logic.
        """
        try:
            args_dict = json.loads(args) if args else {}
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid arguments: {str(e)}"})

        print(f"👻 Virtual Tool Call: {tool_name}({args_dict})")

        result = await backend.generate_response(raw_name, tool_def, args_dict,
                                                 scenario_context=scenario_context)
        return json.dumps(result)

    try:
        return FunctionTool(
            name=tool_name,
            description=tool_desc,
            params_json_schema=params_schema,
            on_invoke_tool=dynamic_run_function,
        )
    except Exception as e:
        print(f"⚠️  Tool {tool_name}: strict schema failed ({e}), retrying with strict_json_schema=False")
        try:
            return FunctionTool(
                name=tool_name,
                description=tool_desc,
                params_json_schema=params_schema,
                on_invoke_tool=dynamic_run_function,
                strict_json_schema=False,
            )
        except Exception as e2:
            print(f"❌ Tool {tool_name}: retry also failed ({e2}), tool will be skipped")
            raise
