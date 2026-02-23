"""
Post-process server quality check agent results.

Reads completed agent trajectories from completion_openai_agent.py output,
extracts the structured JSON quality report from each agent's final message,
loads the original server JSON to attach as metadata, and saves the final
per-server quality results.

No LLM calls are made — this is pure parsing and metadata attachment.

The input *_results.jsonl is NEVER deleted — it is preserved as a sanity-check
record of the full agent trajectories (tool calls, inputs, outputs, reasoning).

Example usage:
    python process_server_quality_check.py \
        --input_file server_quality_check_results.jsonl \
        --output_file server_quality_results.jsonl \
        --server_dir ../mcp_servers/smithery_mcp_servers_0210/

Sanity-check the intermediate trajectories:
    python -c "
    import json
    with open('server_quality_check_results.jsonl') as f:
        for line in f:
            item = json.loads(line)
            print('=== Server:', item['metadata']['server_name'], '===')
            for msg in item['messages']:
                role = msg.get('role')
                if role == 'function':
                    print(f'  [TOOL OUTPUT] {msg.get(\"name\")}: {str(msg.get(\"content\", \"\"))[:200]}')
                elif role == 'assistant' and msg.get('content'):
                    print(f'  [ASSISTANT]: {msg[\"content\"][:300]}')
            print()
    "
"""

import argparse
import glob
import json
import os
import re
import sys
from time import time

# Allow imports from datagen/ (parent directory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from utils import extract_final_response, get_model_abbreviation, load_dataset_from_file, save_dataset


def get_args():
    parser = argparse.ArgumentParser(
        description="Post-process MCP server quality check agent results."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Completed _results.jsonl from server_quality_check_agent.sh.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Final output JSONL path.",
    )
    parser.add_argument(
        "--server_dir",
        type=str,
        required=True,
        help="Original server JSON directory (to load full server_info for metadata).",
    )
    return parser.parse_args()


def load_server_index(server_dir):
    """Build a dict mapping server_id -> full server JSON data."""
    index = {}
    pattern = os.path.join(server_dir, "*.json")
    for file_path in glob.glob(pattern):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            server_id = data.get("server", {}).get("id")
            if server_id:
                index[server_id] = data
        except (json.JSONDecodeError, OSError):
            pass
    return index


def detect_agent_error(messages):
    """
    Detect if the agent failed (as opposed to the server's tools failing).
    Returns (agent_error: bool, error_type: str|None, error_msg: str|None).

    Agent errors are written by completion_openai_agent.py as synthetic assistant
    messages starting with '[ERROR:' or '[UNEXPECTED_ERROR:'.
    """
    last = next(
        (m for m in reversed(messages) if m.get("role") == "assistant" and m.get("content")),
        None,
    )
    if last is None:
        return True, "no_response", "No assistant response in trajectory"

    content = last.get("content", "")
    if content.startswith("[ERROR:") or content.startswith("[UNEXPECTED_ERROR:"):
        return True, classify_agent_error(content), content[:300]

    return False, None, None


def classify_agent_error(error_msg):
    """
    Map a raw agent error string to a short canonical type label.

    Error types:
      timeout                — wrapt timeout killed the worker, or per-item timeout
      max_turns_exceeded     — agent hit the --max_turns limit
      context_length         — prompt exceeded model's token limit
      mcp_invalid_response   — server returned non-MCP-compliant response (Pydantic validation)
      mcp_connection_failed  — all MCP server(s) failed to connect (transient, worth retrying)
      mcp_method_not_found   — server does not implement the requested MCP method
      tool_not_found         — named tool missing from the agent's tool list
      api_error              — HTTP error from the LLM API (4xx/5xx)
      other                  — anything else
    """
    msg = error_msg.lower()
    if "terminated or killed" in msg or "timed out after" in msg:
        return "timeout"
    if "max turns" in msg and "exceeded" in msg:
        return "max_turns_exceeded"
    if "context length" in msg or "maximum context" in msg:
        return "context_length"
    if "calltoolresult" in msg and "field required" in msg:
        return "mcp_invalid_response"
    if "mcp server" in msg and "failed to connect" in msg:
        return "mcp_connection_failed"
    if "method not found" in msg:
        return "mcp_method_not_found"
    if "tool" in msg and "not found" in msg:
        return "tool_not_found"
    if any(code in msg for code in ["error code: 400", "error code: 401", "error code: 403",
                                     "error code: 404", "error code: 429", "error code: 500"]):
        return "api_error"
    return "other"


def extract_json_from_text(text):
    """
    Try to extract a JSON object from agent final message text.
    Handles bare JSON, JSON in markdown code fences, and JSON embedded in text.
    Returns parsed dict or None on failure.
    """
    if not text or not text.strip():
        return None

    # 1. Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2. Try stripping markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Try extracting the outermost {...} block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def parse_quality_report(final_text, tool_names):
    """
    Parse the agent's final JSON quality report.
    Returns list of tool result dicts: [{tool_name, quality, reasoning}].
    Falls back to marking all tools as 'fail' if parsing fails.
    """
    parsed = extract_json_from_text(final_text)

    if parsed and "tool_results" in parsed and isinstance(parsed["tool_results"], list):
        # Validate and clean up results
        results = []
        for entry in parsed["tool_results"]:
            if isinstance(entry, dict) and "tool_name" in entry:
                quality = entry.get("quality", "fail")
                if quality not in ("pass", "fail"):
                    quality = "fail"
                results.append({
                    "tool_name": entry["tool_name"],
                    "quality": quality,
                    "reasoning": str(entry.get("reasoning", "")),
                })
        if results:
            return results

    # Fallback: mark all tools as fail with parse error note
    return [
        {
            "tool_name": name,
            "quality": "fail",
            "reasoning": "Could not parse agent quality report. Raw output: "
                         + (final_text[:200] if final_text else "(empty)"),
        }
        for name in tool_names
    ]


def build_output_item(item, server_data, model_name):
    """Build the final output record for one server."""
    metadata = item.get("metadata", {})
    server_id = metadata.get("server_id", "unknown")
    server_name = metadata.get("server_name", "unknown")
    qualified_name = metadata.get("qualified_name", "")
    tool_names = metadata.get("tool_names", [])

    messages = item.get("messages", [])
    agent_error, agent_error_type, agent_error_msg = detect_agent_error(messages)
    final_text = extract_final_response(messages)
    tool_results = parse_quality_report(final_text, tool_names)

    num_passed = sum(1 for r in tool_results if r["quality"] == "pass")

    return {
        "server_id": server_id,
        "server_name": server_name,
        "qualified_name": qualified_name,
        "agent_error": agent_error,
        "agent_error_type": agent_error_type,
        "agent_error_msg": agent_error_msg,
        "tool_results": tool_results,
        "metadata": {
            "server_info": server_data,
            "model": model_name,
            "timestamp": int(time()),
            "num_tools_checked": len(tool_results),
            "num_tools_passed": num_passed,
        },
    }


def main():
    args = get_args()

    # Load agent results
    print(f"Loading agent results from {args.input_file}...")
    results = load_dataset_from_file(args.input_file)
    if not isinstance(results, list):
        results = [results]
    print(f"Loaded {len(results)} items.")

    # Build server index for metadata attachment
    print(f"Indexing server JSON files from {args.server_dir}...")
    server_index = load_server_index(args.server_dir)
    print(f"Indexed {len(server_index)} servers.")

    # Infer model name from first item's synthetic_data_gen_configs if available
    model_name = "unknown"
    if results:
        configs = results[0].get("metadata", {}).get("synthetic_data_gen_configs", [])
        if configs:
            model_name = configs[-1].get("model", "unknown")

    output_items = []
    error_type_counts = {}

    for item in tqdm(results, desc="Processing results"):
        server_id = item.get("metadata", {}).get("server_id", "unknown")
        server_data = server_index.get(server_id)

        if server_data is None:
            print(f"⚠️  No server JSON found for server_id={server_id}, using empty metadata.")
            server_data = {}

        output_item = build_output_item(item, server_data, model_name)

        if output_item["agent_error"]:
            t = output_item["agent_error_type"] or "other"
            error_type_counts[t] = error_type_counts.get(t, 0) + 1

        output_items.append(output_item)

    total_agent_errors = sum(error_type_counts.values())
    total_tools = sum(o["metadata"]["num_tools_checked"] for o in output_items)
    total_passed = sum(o["metadata"]["num_tools_passed"] for o in output_items)

    print(f"\nSummary:")
    print(f"  Total servers processed: {len(output_items)}")
    print(f"  Agent errors (retryable): {total_agent_errors}")
    if error_type_counts:
        for error_type, count in sorted(error_type_counts.items(), key=lambda x: -x[1]):
            print(f"    {error_type}: {count}")
    print(f"  Agent successes: {len(output_items) - total_agent_errors}")
    if total_tools > 0:
        print(f"  Tools checked: {total_tools}")
        print(f"  Tools passed:  {total_passed} ({100*total_passed//total_tools}%)")

    save_dataset(output_items, args.output_file, convert_to_jsonl=True)
    print(f"\nSaved {len(output_items)} results to {args.output_file}")
    print(f"Note: Intermediate trajectories preserved in {args.input_file}")


if __name__ == "__main__":
    main()
