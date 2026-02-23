"""
Prepare server quality check prompts for use with completion_openai_agent.py.

Reads server JSON files from a directory, filters to servers with no validation error,
and creates one prepared item per server. The agent will test all tools in a single run
and output a structured JSON quality report.

Example usage:
    python prepare_server_quality_check_prompts.py \
        --server_dir ../mcp_servers/smithery_mcp_servers_0210/ \
        --output_file server_quality_check_prepared.jsonl

    # Limit to first 10 servers for testing:
    python prepare_server_quality_check_prompts.py \
        --server_dir ../mcp_servers/smithery_mcp_servers_0210/ \
        --output_file server_quality_check_prepared.jsonl \
        --max_servers 10
"""

import argparse
import glob
import json
import os
import sys

# Allow imports from datagen/ (parent directory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jinja2 import Environment, FileSystemLoader, exceptions
from tqdm import tqdm

from utils import save_dataset


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare MCP server quality check prompts."
    )
    parser.add_argument(
        "--server_dir",
        type=str,
        required=True,
        help="Directory containing server JSON files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output prepared JSONL file path.",
    )
    parser.add_argument(
        "--max_servers",
        type=int,
        default=None,
        help="Optional cap on number of servers to prepare (useful for testing).",
    )
    return parser.parse_args()


def load_prompt_template():
    """Load the server quality check prompt template."""
    prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts")
    env = Environment(loader=FileSystemLoader(prompts_dir))
    try:
        return env.get_template("server_quality_check.md").render()
    except exceptions.TemplateNotFound:
        raise FileNotFoundError(
            f"server_quality_check.md not found in {prompts_dir}"
        )


def build_tool_list_str(tools):
    """Build a formatted tool list string for the prompt."""
    lines = []
    for tool in tools:
        name = tool.get("name", "unknown")
        description = tool.get("description", "No description available.")
        input_schema = tool.get("inputSchema", {})
        schema_str = json.dumps(input_schema, ensure_ascii=False)
        lines.append(f"- **{name}**: {description}\n  Input schema: {schema_str}")
    return "\n".join(lines)


def build_prepared_item(server_data, file_path, row_id, template):
    """Build a single prepared item from a server JSON."""
    server = server_data.get("server", {})
    server_id = server.get("id", "unknown")
    server_name = server.get("displayName", server.get("qualifiedName", "Unknown Server"))
    qualified_name = server.get("qualifiedName", "")
    tools = server.get("tools", [])

    tool_list_str = build_tool_list_str(tools)

    # Fill in prompt template
    content = template.replace("{SERVER_NAME}", server_name)
    content = content.replace("{TOOL_LIST}", tool_list_str)

    # Relative path from datagen/ to the server JSON file (used by construct_mcp_url_from_source)
    # completion_openai_agent.py resolves source_file_path relative to its own location (datagen/)
    datagen_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel_path = os.path.relpath(os.path.abspath(file_path), datagen_dir)

    return {
        "messages": [
            {"role": "user", "content": content}
        ],
        "metadata": {
            "prompt_id": server_id,
            "row_id": row_id,
            "server_id": server_id,
            "server_name": server_name,
            "qualified_name": qualified_name,
            "tool_names": [t.get("name", "unknown") for t in tools],
            "mcp_servers": [
                {
                    "server_name": server_name,
                    "source_file_path": rel_path,
                }
            ],
        },
    }


def main():
    args = get_args()

    # Load prompt template
    template = load_prompt_template()

    # Glob all JSON files in the server directory
    pattern = os.path.join(args.server_dir, "*.json")
    all_files = sorted(glob.glob(pattern))

    if not all_files:
        print(f"No JSON files found in {args.server_dir}")
        return

    print(f"Found {len(all_files)} server JSON files in {args.server_dir}")

    prepared_items = []
    skipped_no_tools = 0
    skipped_validation_error = 0
    row_id = 0

    for file_path in tqdm(all_files, desc="Preparing servers"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️ Skipping {file_path}: {e}")
            continue

        server = data.get("server", {})

        # Filter: only process servers with null validation_error
        if server.get("validation_error") is not None:
            skipped_validation_error += 1
            continue

        # Skip servers with no tools
        tools = server.get("tools", [])
        if not tools:
            skipped_no_tools += 1
            continue

        item = build_prepared_item(data, file_path, row_id, template)
        prepared_items.append(item)
        row_id += 1

        if args.max_servers is not None and len(prepared_items) >= args.max_servers:
            print(f"Reached --max_servers limit of {args.max_servers}. Stopping.")
            break

    print(f"\nSummary:")
    print(f"  Prepared:               {len(prepared_items)}")
    print(f"  Skipped (validation error): {skipped_validation_error}")
    print(f"  Skipped (no tools):     {skipped_no_tools}")

    if not prepared_items:
        print("No items to save.")
        return

    save_dataset(prepared_items, args.output_file, convert_to_jsonl=True)
    print(f"\nSaved {len(prepared_items)} prepared items to {args.output_file}")


if __name__ == "__main__":
    main()
