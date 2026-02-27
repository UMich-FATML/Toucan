"""
Prepare per-tool retry prompts for servers with retryable agent failures.

Reads server_quality_results.jsonl (output of process_server_quality_check.py),
filters to servers with retryable agent errors, and creates one prepared item
per tool per server. Each prompt tests only a single tool, avoiding the context
and turn-count issues that cause per-server runs to fail.

Two modes:
  server (default): filters servers where agent_error=True and agent_error_type
                    is not in --skip_error_types. Re-queues ALL tools for those servers.
  tool:             filters individual tool results where quality="fail" and the
                    reasoning contains one of --tool_error_patterns. Use this to
                    retry specific tool-level failures (e.g. mcp_connection_failed)
                    on servers that otherwise processed successfully.

Example usage:
    # Server-level retry (default):
    python prepare_tool_retry_prompts.py \\
        --quality_results_file server_quality_results.jsonl \\
        --server_dir ../mcp_servers/smithery_mcp_servers_0210/ \\
        --output_file tool_retry_prepared.jsonl

    # Tool-level retry (connection failures only):
    python prepare_tool_retry_prompts.py \\
        --quality_results_file server_quality_results_merged.jsonl \\
        --server_dir ../mcp_servers/smithery_mcp_servers_0210/ \\
        --output_file tool_retry2_prepared.jsonl \\
        --mode tool \\
        --tool_error_patterns "mcp_connection_failed"
"""

import argparse
import json
import os
import sys

# Allow imports from datagen/ (parent directory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from utils import load_dataset_from_file, save_dataset

SKIP_BY_DEFAULT = {"context_length", "mcp_invalid_response", "mcp_method_not_found", "timeout"}


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare per-tool retry prompts for failed servers."
    )
    parser.add_argument(
        "--quality_results_file",
        type=str,
        required=True,
        help="server_quality_results.jsonl from process_server_quality_check.py.",
    )
    parser.add_argument(
        "--server_dir",
        type=str,
        required=True,
        help="Directory containing original server JSON files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output prepared JSONL path.",
    )
    parser.add_argument(
        "--skip_error_types",
        nargs="*",
        default=list(SKIP_BY_DEFAULT),
        help=(
            f"Agent error types to skip (default: {sorted(SKIP_BY_DEFAULT)}). "
            "Pass an empty list to retry all error types."
        ),
    )
    parser.add_argument(
        "--max_tools",
        type=int,
        default=None,
        help="Optional cap on total number of tool prompts to prepare (useful for testing).",
    )
    parser.add_argument(
        "--mode",
        choices=["server", "tool"],
        default="server",
        help=(
            "server (default): retry all tools for servers where agent_error=True. "
            "tool: retry specific tools whose result reasoning matches --tool_error_patterns."
        ),
    )
    parser.add_argument(
        "--tool_error_patterns",
        nargs="+",
        default=["mcp_connection_failed"],
        help=(
            "Substrings to match in tool result reasoning when --mode=tool. "
            "Defaults to ['mcp_connection_failed']."
        ),
    )
    return parser.parse_args()


def load_prompt_template():
    """Load the single-tool retry prompt template."""
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts", "tool_retry.md")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def build_tool_entry_str(tool):
    """Format a single tool's name, description, and schema for the prompt."""
    name = tool.get("name", "unknown")
    description = tool.get("description", "No description available.")
    input_schema = tool.get("inputSchema", {})
    schema_str = json.dumps(input_schema, ensure_ascii=False)
    return f"**{name}**: {description}\n  Input schema: {schema_str}"


def build_prepared_item(server_id, server_name, qualified_name, tool, rel_path, row_id, template):
    """Build a single prepared item for one tool on one server."""
    tool_name = tool.get("name", "unknown")
    tool_entry_str = build_tool_entry_str(tool)

    content = template.replace("{SERVER_NAME}", server_name)
    content = content.replace("{TOOL_ENTRY}", tool_entry_str)

    return {
        "messages": [{"role": "user", "content": content}],
        "metadata": {
            "prompt_id": f"{server_id}__{tool_name}",
            "row_id": row_id,
            "server_id": server_id,
            "server_name": server_name,
            "qualified_name": qualified_name,
            "tool_names": [tool_name],
            "mcp_servers": [
                {
                    "server_name": server_name,
                    "source_file_path": rel_path,
                }
            ],
        },
    }


def tool_result_matches(tool_result, patterns):
    """Return True if the tool result reasoning contains any of the given patterns."""
    reasoning = tool_result.get("reasoning", "").lower()
    return any(p.lower() in reasoning for p in patterns)


def collect_server_mode(quality_results, skip_types):
    """Collect (server_result, tools_to_retry) pairs using server-level agent_error filtering."""
    retryable = [
        r for r in quality_results
        if r.get("agent_error") and r.get("agent_error_type") not in skip_types
    ]
    # In server mode, retry all tools for the server
    return [(r, None) for r in retryable]  # None = all tools


def collect_tool_mode(quality_results, patterns):
    """Collect (server_result, tools_to_retry) pairs by scanning individual tool results."""
    pairs = []
    for r in quality_results:
        matched_tool_names = {
            tr["tool_name"]
            for tr in r.get("tool_results", [])
            if tool_result_matches(tr, patterns)
        }
        if matched_tool_names:
            pairs.append((r, matched_tool_names))
    return pairs


def main():
    args = get_args()
    skip_types = set(args.skip_error_types or [])

    print(f"Loading quality results from {args.quality_results_file}...")
    quality_results = load_dataset_from_file(args.quality_results_file)
    if not isinstance(quality_results, list):
        quality_results = [quality_results]
    print(f"Loaded {len(quality_results)} server results.")

    if args.mode == "server":
        pairs = collect_server_mode(quality_results, skip_types)
        print(f"Retryable servers (skipping {sorted(skip_types)}): {len(pairs)}")
    else:
        pairs = collect_tool_mode(quality_results, args.tool_error_patterns)
        print(f"Servers with tools matching {args.tool_error_patterns}: {len(pairs)}")

    template = load_prompt_template()
    # completion_openai_agent.py resolves source_file_path relative to datagen/
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    prepared_items = []
    skipped_no_server_file = 0
    skipped_no_tools = 0
    row_id = 0

    for result, tool_name_filter in tqdm(pairs, desc="Preparing tool prompts"):
        server_id = result.get("server_id", "unknown")
        server_name = result.get("server_name", "unknown")
        qualified_name = result.get("qualified_name", "")

        # Get full tool definitions from embedded server_info
        server_info = result.get("metadata", {}).get("server_info", {})
        all_tools = server_info.get("server", {}).get("tools", [])

        # In tool mode, only retry the specific tools that matched
        if tool_name_filter is not None:
            tools = [t for t in all_tools if t.get("name") in tool_name_filter]
        else:
            tools = all_tools

        if not tools:
            skipped_no_tools += 1
            continue

        # Verify the server JSON file exists (needed for MCP URL construction)
        server_json_path = os.path.join(args.server_dir, f"{server_id}.json")
        if not os.path.exists(server_json_path):
            print(f"⚠️  Server file not found: {server_json_path}, skipping {server_name}")
            skipped_no_server_file += 1
            continue

        rel_path = os.path.relpath(os.path.abspath(server_json_path), script_dir)

        for tool in tools:
            item = build_prepared_item(
                server_id, server_name, qualified_name, tool, rel_path, row_id, template
            )
            prepared_items.append(item)
            row_id += 1

            if args.max_tools is not None and len(prepared_items) >= args.max_tools:
                print(f"Reached --max_tools limit of {args.max_tools}. Stopping.")
                break

        if args.max_tools is not None and len(prepared_items) >= args.max_tools:
            break

    print(f"\nSummary:")
    print(f"  Servers matched:           {len(pairs)}")
    print(f"  Skipped (no server file):  {skipped_no_server_file}")
    print(f"  Skipped (no tools):        {skipped_no_tools}")
    print(f"  Tool prompts prepared:     {len(prepared_items)}")

    if not prepared_items:
        print("No items to save.")
        return

    save_dataset(prepared_items, args.output_file, convert_to_jsonl=True)
    print(f"\nSaved {len(prepared_items)} tool prompts to {args.output_file}")


if __name__ == "__main__":
    main()
