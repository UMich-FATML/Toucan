import json
from typing import Dict, Any, List, Optional
from openai import AsyncClient

from agents import FunctionTool, RunContextWrapper


VIRTUAL_TOOL_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "error": {"type": "string"},
        "response": {
            "anyOf": [
                {"type": "string"},
                {"type": "object", "additionalProperties": True},
                {"type": "array"},
                {"type": "number"},
                {"type": "integer"},
                {"type": "boolean"},
                {"type": "null"},
            ]
        },
    },
    "required": ["error", "response"],
    "additionalProperties": False,
}


VIRTUAL_SCENARIO_MASTER_SYSTEM_PROMPT = """You are the Scenario Master for tool-calling scenarios, similar to a DnD dungeon master.
Your job is to co-create and play out realistic tool-calling scenarios with the user over multiple turns.
When the user provides tool simulation requests, treat them as scenario events and simulate the tool responses as the scenario unfolds.

You must always return exactly one JSON object in this schema:
{
  "error": "",
  "response": ""
}

You will receive tool metadata and tool input for each event.
Follow this process exactly:

STEP 1: STRICT VALIDATION
Special case: if the tool documentation contains no inputSchema, or inputSchema has no properties
(properties missing or empty {}), skip validation and go to Step 2.

Otherwise validate tool_input against input_schema:
1) Missing required arguments: all required keys must be present.
2) Hallucinated arguments: do not allow unknown keys not listed in properties.
3) Type mismatches: values must match declared types and enum constraints.

STEP 2: RESPONSE GENERATION
If validation fails:
- Halt simulation for that call.
- Put a concise, specific reason in "error".
- Set "response" to an empty string.

If validation succeeds:
- Keep "error" as an empty string.
- Populate "response" with meaningful, practical content that matches the tool's intended functionality.
- Maintain JSON integrity.
- If information is incomplete, still generate a useful, realistic response and fabricate plausible values when needed.

Realism requirements:
- Avoid obvious placeholders like Jane Doe, John Smith, Acme Corp, 123 Main St, example@email.com.
- Use plausible names, IDs, addresses, and domain details.
- Stay consistent with scenario context from previous conversation turns.

Context usage:
- Incorporate scenario context from the conversation history.
- Incorporate previous simulated tool-call turns to maintain continuity.
- Use the current server and tool details as ground truth for this specific call.

Output constraints:
- Return only the JSON object; no extra prose or markdown.
- Ensure the object is parseable JSON.
"""


def load_prompt_template(template_path):
  """Load the prompt template from file."""
  with open(template_path, 'r', encoding='utf-8') as f:
    return f.read()


def _content_to_text(content: Any) -> str:
    """Normalize chat message content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
                elif "text" in part:
                    chunks.append(str(part.get("text", "")))
                else:
                    chunks.append(json.dumps(part, ensure_ascii=False))
            else:
                chunks.append(str(part))
        return "\n".join([c for c in chunks if c])
    if isinstance(content, (dict, tuple, set)):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _format_non_conversational_message(role: str, name: str, content_text: str) -> str:
    """Map non user/assistant roles into assistant narration."""
    if role == "system":
        return f"Scenario setup:\n{content_text}"
    if role == "tool":
        tool_label = name or "tool"
        return f"Scenario update from {tool_label}:\n{content_text}"
    if name:
        return f"Scenario update ({role}, {name}):\n{content_text}"
    return f"Scenario update ({role}):\n{content_text}"


def normalize_scenario_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Keep a user/scenario-master conversation style:
    - Keep user/assistant roles as-is.
    - Convert other roles to assistant narration.
    """
    normalized: List[Dict[str, str]] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        if not role:
            continue
        content_text = _content_to_text(msg.get("content"))
        if not content_text:
            continue

        if role in {"user", "assistant"}:
            normalized.append({"role": role, "content": content_text})
            continue

        name = str(msg.get("name", "")).strip()
        normalized.append(
            {
                "role": "assistant",
                "content": _format_non_conversational_message(role, name, content_text),
            }
        )
    return normalized


class VirtualToolBackend:
    """
    Manages the LLM calls for generating virtual tool responses.
    """
    def __init__(self, client: AsyncClient, model_path: str):
        self.client = client
        self.model = model_path

    def build_tool_simulation_request(self, tool_doc: Dict, tool_args: Dict,
                                      scenario_context: Optional[Dict] = None) -> str:
        """Render the user message template for a single tool simulation request."""
        template = load_prompt_template('./prompts/virtual_toucan.md')

        tool_name = tool_doc.get("name", "")
        tool_description = tool_doc.get("description", "")
        tool_schema = tool_doc.get("inputSchema", tool_doc.get("input_schema", {}))

        server_id = ""
        server_name = ""
        server_description = ""
        if scenario_context:
            server_id = scenario_context.get("server_id", "") or ""
            server_name = scenario_context.get("server_name", "") or ""
            server_description = scenario_context.get("server_description", "") or ""

        replacements = {
            "{SERVER_ID}": str(server_id),
            "{SERVER_NAME}": str(server_name),
            "{SERVER_DESCRIPTION}": str(server_description),
            "{TOOL_NAME}": str(tool_name),
            "{TOOL_DESCRIPTION}": str(tool_description),
            "{TOOL_INPUT_SCHEMA_JSON}": json.dumps(tool_schema, ensure_ascii=False, indent=2),
            "{TOOL_ARGS_JSON}": json.dumps(tool_args, ensure_ascii=False, indent=2),
        }

        rendered = template
        for key, value in replacements.items():
            rendered = rendered.replace(key, value)
        return rendered

    async def generate_response(self, tool_name: str, tool_doc: Dict, tool_args: Dict,
                                scenario_context: Optional[Dict] = None,
                                current_request_message: Optional[str] = None) -> Dict:
        """
        Hits the LLM to hallucinate a response for the tool.
        If scenario_context is provided, includes conversation context and
        prior simulation turns to keep scenario continuity.
        """
        conversation_history: List[Dict[str, Any]] = []
        prior_tool_sim_messages: List[Dict[str, Any]] = []
        if scenario_context:
            conversation_history = scenario_context.get('conversation_history', []) or []
            prior_tool_sim_messages = scenario_context.get("tool_simulation_messages", []) or []

        if current_request_message is None:
            current_request_message = self.build_tool_simulation_request(
                tool_doc=tool_doc,
                tool_args=tool_args,
                scenario_context=scenario_context,
            )

        messages = [{"role": "system", "content": VIRTUAL_SCENARIO_MASTER_SYSTEM_PROMPT}]
        messages.extend(normalize_scenario_messages(conversation_history))
        messages.extend(normalize_scenario_messages(prior_tool_sim_messages))
        messages.append({"role": "user", "content": current_request_message})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "virtual_tool_response",
                        "strict": True,
                        "schema": VIRTUAL_TOOL_OUTPUT_JSON_SCHEMA,
                    },
                },
            )

            content = response.choices[0].message.content

            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = {"error": "Failed to parse JSON output", "response": content}

            return result

        except Exception as e:
            print(f"⚠️ Virtual Tool Generation Failed for {tool_name}: {e}")
            return {"error": str(e), "response": ""}


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

        request_message = backend.build_tool_simulation_request(
            tool_doc=tool_def,
            tool_args=args_dict,
            scenario_context=scenario_context,
        )

        result = await backend.generate_response(
            raw_name,
            tool_def,
            args_dict,
            scenario_context=scenario_context,
            current_request_message=request_message,
        )

        shared_tool_history = None
        if scenario_context:
            shared_tool_history = scenario_context.get("tool_simulation_messages")
        if isinstance(shared_tool_history, list):
            shared_tool_history.extend(
                [
                    {"role": "user", "content": request_message},
                    {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)},
                ]
            )

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
