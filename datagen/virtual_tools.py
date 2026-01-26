import json
import asyncio
from typing import Dict, Any, List, Callable
from openai import AsyncClient
from pydantic import create_model, Field, BaseModel

# 1. UPDATED IMPORTS: Use FunctionTool and RunContextWrapper
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

    async def generate_response(self, tool_name: str, tool_doc: Dict, tool_args: Dict) -> Dict:
        """
        Hits the LLM to hallucinate a response for the tool.
        """
        user_prompt = (
            f"Tool Documentation: {json.dumps(tool_doc)}\n\n"
            f"Generate a realistic JSON response for the following input:\n"
            f"{json.dumps(tool_args)}"
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

def create_dynamic_virtual_tool(tool_def: Dict, backend: VirtualToolBackend):
    """
    Dynamically creates a FunctionTool compatible with openai-agents.
    """
    tool_name = tool_def.get('name')
    tool_desc = tool_def.get('description', '')
    input_schema = tool_def.get('input_schema', {})
    
    # 1. Map types
    type_map = {
        'string': str, 'integer': int, 'number': float, 
        'boolean': bool, 'array': list, 'object': dict
    }

    # 2. Build fields for Pydantic
    fields = {}
    properties = input_schema.get('properties', {})
    required = set(input_schema.get('required', []))

    for param_name, param_info in properties.items():
        param_type = type_map.get(param_info.get('type', 'string'), str)
        param_desc = param_info.get('description', '')
        
        if param_name in required:
            default = ... 
        else:
            default = param_info.get('default', None)

        fields[param_name] = (param_type, Field(default=default, description=param_desc))

    # 3. Create Pydantic Model
    DynamicParams = create_model(f"{tool_name}_Args", **fields)

    # 4. Define the execution function
    # NOTE: The signature must match what FunctionTool expects: (ctx, args_string)
    async def dynamic_run_function(ctx: RunContextWrapper[Any], args: str) -> str:
        """
        Executes the virtual tool logic.
        """
        # Parse the JSON string arguments into our Pydantic model
        try:
            params = DynamicParams.model_validate_json(args)
            args_dict = params.model_dump()
        except Exception as e:
            return json.dumps({"error": f"Invalid arguments: {str(e)}"})

        print(f"👻 Virtual Tool Call: {tool_name}({args_dict})")
        
        # Call backend
        result = await backend.generate_response(tool_name, tool_def, args_dict)
        
        # FunctionTool expects a string return
        return json.dumps(result)

    # 5. Return the FunctionTool Object
    return FunctionTool(
        name=tool_name,
        description=tool_desc,
        params_json_schema=DynamicParams.model_json_schema(),
        on_invoke_tool=dynamic_run_function
    )