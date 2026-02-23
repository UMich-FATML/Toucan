"""
Merge per-tool retry results back into server_quality_results.jsonl.

Reads completed agent trajectories from tool_retry_agent.sh output, extracts
the single-tool JSON quality report from each item, then merges per-tool results
into the original server_quality_results.jsonl by replacing tool_results for
servers that were retried. Servers not covered by the retry pass through unchanged.

No LLM calls are made — this is pure parsing and merging.

The input *_results.jsonl is NEVER deleted — it is preserved as a record of the
full per-tool agent trajectories.

Example usage:
    python process_tool_retry_results.py \\
        --input_file tool_retry_results.jsonl \\
        --quality_results_file server_quality_results.jsonl \\
        --output_file server_quality_results_merged.jsonl
"""

import argparse
import os
import sys
from collections import defaultdict
from time import time

# Allow imports from datagen/ (parent directory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from utils import extract_final_response, load_dataset_from_file, save_dataset
from process_server_quality_check import detect_agent_error, parse_quality_report


def get_args():
    parser = argparse.ArgumentParser(
        description="Merge per-tool retry results into server_quality_results.jsonl."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Completed tool retry _results.jsonl from tool_retry_agent.sh.",
    )
    parser.add_argument(
        "--quality_results_file",
        type=str,
        required=True,
        help="Original server_quality_results.jsonl to merge into.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output merged JSONL path.",
    )
    return parser.parse_args()


def extract_tool_retry_result(item):
    """
    Extract the per-tool quality result from a single tool retry agent trajectory.
    Returns (server_id, tool_name, tool_result_dict).

    If the agent itself failed (timeout, max turns, etc.), marks the tool as fail
    with the agent error in the reasoning so the failure mode is visible in output.
    """
    metadata = item.get("metadata", {})
    server_id = metadata.get("server_id", "unknown")
    tool_names = metadata.get("tool_names", [])
    tool_name = tool_names[0] if tool_names else "unknown"

    messages = item.get("messages", [])
    agent_error, agent_error_type, agent_error_msg = detect_agent_error(messages)

    if agent_error:
        tool_result = {
            "tool_name": tool_name,
            "quality": "fail",
            "reasoning": f"Agent error ({agent_error_type}): {agent_error_msg or 'unknown'}",
        }
    else:
        final_text = extract_final_response(messages)
        results = parse_quality_report(final_text, [tool_name])
        tool_result = results[0] if results else {
            "tool_name": tool_name,
            "quality": "fail",
            "reasoning": "No quality report parsed from agent output.",
        }

    return server_id, tool_name, tool_result


def main():
    args = get_args()

    # Load original quality results indexed by server_id
    print(f"Loading original quality results from {args.quality_results_file}...")
    original_results = load_dataset_from_file(args.quality_results_file)
    if not isinstance(original_results, list):
        original_results = [original_results]
    original_by_server = {r["server_id"]: r for r in original_results}
    print(f"Loaded {len(original_results)} original server entries.")

    # Load and parse tool retry agent results
    print(f"Loading tool retry results from {args.input_file}...")
    retry_items = load_dataset_from_file(args.input_file)
    if not isinstance(retry_items, list):
        retry_items = [retry_items]
    print(f"Loaded {len(retry_items)} tool retry items.")

    # Group retry results by server_id: {server_id -> {tool_name -> tool_result}}
    retry_by_server = defaultdict(dict)
    retry_agent_errors = 0
    for item in tqdm(retry_items, desc="Parsing retry results"):
        server_id, tool_name, tool_result = extract_tool_retry_result(item)
        retry_by_server[server_id][tool_name] = tool_result
        if tool_result["quality"] == "fail" and tool_result["reasoning"].startswith("Agent error"):
            retry_agent_errors += 1

    print(f"Retry results cover {len(retry_by_server)} unique servers.")
    if retry_agent_errors:
        print(f"  ({retry_agent_errors} individual tool retries also hit agent errors)")

    # Merge: update entries for retried servers, pass through the rest
    output_items = []
    servers_updated = 0
    tools_updated = 0

    for server_id, original in original_by_server.items():
        if server_id not in retry_by_server:
            output_items.append(original)
            continue

        retry_tools = retry_by_server[server_id]
        old_tool_results = original.get("tool_results", [])
        old_by_name = {r["tool_name"]: r for r in old_tool_results}

        # Retry result takes precedence for each tool; keep originals for any not retried
        new_tool_results = []
        for tool_name, old_result in old_by_name.items():
            if tool_name in retry_tools:
                new_tool_results.append(retry_tools[tool_name])
                tools_updated += 1
            else:
                new_tool_results.append(old_result)

        # Include tools from retry not present in original (edge case)
        for tool_name, result in retry_tools.items():
            if tool_name not in old_by_name:
                new_tool_results.append(result)
                tools_updated += 1

        num_passed = sum(1 for r in new_tool_results if r["quality"] == "pass")

        updated = dict(original)
        updated["agent_error"] = False
        updated["agent_error_type"] = None
        updated["agent_error_msg"] = None
        updated["tool_results"] = new_tool_results
        updated["metadata"] = dict(original.get("metadata", {}))
        updated["metadata"]["num_tools_checked"] = len(new_tool_results)
        updated["metadata"]["num_tools_passed"] = num_passed
        updated["metadata"]["retry_timestamp"] = int(time())

        output_items.append(updated)
        servers_updated += 1

    total_tools = sum(o["metadata"]["num_tools_checked"] for o in output_items)
    total_passed = sum(o["metadata"]["num_tools_passed"] for o in output_items)

    print(f"\nSummary:")
    print(f"  Total servers in output:   {len(output_items)}")
    print(f"  Servers updated by retry:  {servers_updated}")
    print(f"  Tools updated:             {tools_updated}")
    if total_tools > 0:
        print(f"  Tools checked (total):     {total_tools}")
        print(f"  Tools passed  (total):     {total_passed} ({100*total_passed//total_tools}%)")

    save_dataset(output_items, args.output_file, convert_to_jsonl=True)
    print(f"\nSaved {len(output_items)} entries to {args.output_file}")
    print(f"Note: Intermediate retry trajectories preserved in {args.input_file}")


if __name__ == "__main__":
    main()
