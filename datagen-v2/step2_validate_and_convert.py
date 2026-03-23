#!/usr/bin/env python3
"""
Step 2 Alternative: Validate and Convert

Lightweight replacement for step 2.1-2.3 that skips LLM quality checks.
Takes _3sanitized.jsonl from step 1.3, validates entries, converts to the
prepared format for the completion agent, and extracts an answer key file
for step 4.1 evaluation.

Example Usage:
    python step2_validate_and_convert.py --input_file ../data/.../foo_3sanitized.jsonl
    python step2_validate_and_convert.py --input_file ../data/.../foo_3sanitized.jsonl --require_outputs
"""

import os
import argparse
import json
from tqdm import tqdm
from utils import clean_json_object, create_preview_json


def get_args():
    parser = argparse.ArgumentParser(
        description="Step 2 Alternative: Validate step 1.3 output and convert to prepared + answer key formats."
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to _3sanitized.jsonl from step 1.3.")
    parser.add_argument("--require_outputs", action="store_true",
                        help="Skip entries that lack target_tools_with_outputs. "
                             "Without this flag, entries without outputs are still included "
                             "in the prepared file but omitted from the answer key.")
    return parser.parse_args()


def filter_metadata_by_target_tools(metadata, target_tools_str):
    """
    Filter metadata to only include MCP servers that provide the tools
    mentioned in target_tools. Only applies when allocation strategy is
    random_featured. Replicates step1.3 logic.
    """
    if not (isinstance(target_tools_str, str) and target_tools_str.strip()):
        return metadata

    target_tools_raw = [t.strip() for t in target_tools_str.split(',') if t.strip()]
    if not target_tools_raw:
        return metadata

    if metadata.get("question_gen_args", {}).get("multi_server_allocation_strategy", "") != "random_featured":
        return metadata

    if "mcp_servers" not in metadata:
        return metadata

    filtered_metadata = metadata.copy()
    servers = filtered_metadata["mcp_servers"]

    server_tool_combos = set()
    for tool_entry in target_tools_raw:
        if '::' in tool_entry:
            server_name, tool_name = tool_entry.split('::', 1)
            server_tool_combos.add((server_name.strip(), tool_name.strip()))
        else:
            return metadata  # Can't filter without server::tool format

    filtered_servers = []
    for server_info in servers:
        server_name = server_info.get("server_name", "Unknown Server")
        remote_response = server_info.get("remote_server_response", {})
        server_tools = remote_response.get("tools", [])

        for tool in server_tools:
            if (server_name, tool.get("name", "")) in server_tool_combos:
                filtered_servers.append(server_info)
                break

    filtered_metadata["mcp_servers"] = filtered_servers
    if "server_count" in filtered_metadata:
        filtered_metadata["server_count"] = len(filtered_servers)

    return filtered_metadata


def validate_entry(data, line_num, require_outputs):
    """
    Validate a single entry from _3sanitized.jsonl.
    Returns (is_valid, reason) tuple.
    """
    # Check question
    question = data.get("question", "")
    if not question or not isinstance(question, str) or len(question.strip()) < 10:
        return False, "missing_or_short_question"

    # Check target_tools (accepts list from step1.3 or string)
    target_tools = data.get("target_tools")
    target_tools_str = data.get("target_tools_str", "")
    if isinstance(target_tools, list):
        if not target_tools:
            return False, "missing_target_tools"
    elif isinstance(target_tools, str):
        if not target_tools.strip():
            return False, "missing_target_tools"
    elif not (target_tools_str and target_tools_str.strip()):
        return False, "missing_target_tools"

    # Check metadata
    metadata = data.get("metadata")
    if not metadata or not isinstance(metadata, dict):
        return False, "missing_metadata"

    # Check mcp_servers
    mcp_servers = metadata.get("mcp_servers", [])
    if not mcp_servers or not isinstance(mcp_servers, list) or len(mcp_servers) == 0:
        return False, "missing_mcp_servers"

    # Check target_tools_with_outputs if required
    if require_outputs:
        outputs = data.get("target_tools_with_outputs")
        if not outputs or not isinstance(outputs, list) or len(outputs) == 0:
            # Fall back to target_tools items with embedded output (new format)
            ttools = data.get("target_tools")
            if isinstance(ttools, list) and ttools:
                outputs = ttools
            else:
                return False, "missing_target_tools_with_outputs"
        for i, tool_entry in enumerate(outputs):
            if not isinstance(tool_entry, dict):
                return False, f"invalid_tool_entry_{i}"
            if not tool_entry.get("tool"):
                return False, f"missing_tool_name_in_entry_{i}"
            if "arguments" not in tool_entry:
                return False, f"missing_arguments_in_entry_{i}"

    return True, "valid"


def prepare_entry(data):
    """
    Convert a _3sanitized entry to the prepared format expected by
    completion_openai_agent.py. Replicates step 1.3's prepare_questions() logic.
    """
    metadata = data.get("metadata", {})
    target_tools = data.get("target_tools")
    target_tools_str = data.get("target_tools_str", "")
    if isinstance(target_tools, list) and not target_tools_str:
        target_tools_str = ", ".join(
            f"{t.get('server', '')}::{t.get('tool', '')}"
            for t in target_tools
            if isinstance(t, dict) and t.get("server") and t.get("tool")
        )
    elif isinstance(target_tools, str):
        target_tools_str = target_tools
    filtered_metadata = filter_metadata_by_target_tools(metadata, target_tools_str)

    result = {
        "messages": [
            {
                "role": "user",
                "content": data["question"]
            }
        ],
        "metadata": {
            **filtered_metadata,
            "target_tools": target_tools,
            "target_tools_str": target_tools_str,
            "question": data["question"],
            "min_distance": data.get("min_distance"),
            "duplicate_count": data.get("duplicate_count", 0),
            "min_similar_row_id": data.get("min_similar_row_id"),
        }
    }

    if data.get("tool_analysis"):
        result["metadata"]["tool_analysis"] = data["tool_analysis"]

    if data.get("cross_tool_workflow"):
        result["metadata"]["cross_tool_workflow"] = data["cross_tool_workflow"]

    if data.get("target_tools_with_outputs"):
        result["metadata"]["target_tools_with_outputs"] = data["target_tools_with_outputs"]

    if data.get("withheld_info"):
        result["metadata"]["withheld_info"] = data["withheld_info"]

    if data.get("target_followup_questions"):
        result["metadata"]["target_followup_questions"] = data["target_followup_questions"]

    return clean_json_object(result)


def extract_answer_key(data):
    """
    Extract an answer key entry from target_tools_with_outputs.
    Returns None if no outputs are present.
    """
    outputs = data.get("target_tools_with_outputs")
    if not outputs or not isinstance(outputs, list):
        # New format: 'output' is embedded directly in each target_tools item
        target_tools = data.get("target_tools")
        if isinstance(target_tools, list) and target_tools:
            outputs = target_tools
        else:
            return None

    answer_key = []
    for tool_entry in outputs:
        if not isinstance(tool_entry, dict):
            continue
        tool_name = tool_entry.get("tool", "")
        arguments = tool_entry.get("arguments", {})
        if tool_name:
            answer_key.append({"tool": tool_name, "arguments": arguments})

    if not answer_key:
        return None

    row_id = data.get("metadata", {}).get("row_id")
    result = {
        "metadata": {"row_id": row_id},
        "answer_key": answer_key,
    }

    if data.get("withheld_info"):
        result["metadata"]["withheld_info"] = data["withheld_info"]

    if data.get("target_followup_questions"):
        result["metadata"]["target_followup_questions"] = data["target_followup_questions"]

    return result


def main():
    args = get_args()
    print(f"Step 2 Alternative: Validate and Convert\nArguments:\n{args}")

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    # Determine output paths
    input_dir = os.path.dirname(args.input_file)
    input_basename = os.path.basename(args.input_file)
    base_name = input_basename.replace("_3sanitized.jsonl", "")

    prepared_output = os.path.join(input_dir, f"{base_name}_validated_prepared.jsonl")
    answer_key_output = os.path.join(input_dir, f"{base_name}_answer_key.jsonl")
    prepared_preview = os.path.join(input_dir, f"preview_{base_name}_validated_prepared.json")
    answer_key_preview = os.path.join(input_dir, f"preview_{base_name}_answer_key.json")

    # Stats
    stats = {
        "total_lines": 0,
        "valid_entries": 0,
        "answer_keys_generated": 0,
        "skipped": {},
    }

    with (
        open(args.input_file, 'r', encoding='utf-8') as f_in,
        open(prepared_output, 'w', encoding='utf-8') as f_prepared,
        open(answer_key_output, 'w', encoding='utf-8') as f_answer_key,
    ):
        for line_num, line in enumerate(tqdm(f_in, desc="Validating and converting"), 1):
            stats["total_lines"] += 1

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                stats["skipped"]["json_decode_error"] = stats["skipped"].get("json_decode_error", 0) + 1
                print(f"Line {line_num}: JSON decode error: {e}")
                continue

            is_valid, reason = validate_entry(data, line_num, args.require_outputs)
            if not is_valid:
                stats["skipped"][reason] = stats["skipped"].get(reason, 0) + 1
                continue

            # Prepare entry for completion agent
            prepared = prepare_entry(data)
            f_prepared.write(json.dumps(prepared, ensure_ascii=False) + '\n')
            stats["valid_entries"] += 1

            # Extract answer key if outputs are available
            answer_key = extract_answer_key(data)
            if answer_key:
                f_answer_key.write(json.dumps(answer_key, ensure_ascii=False) + '\n')
                stats["answer_keys_generated"] += 1

    # Create previews
    create_preview_json(prepared_output, prepared_preview)
    create_preview_json(answer_key_output, answer_key_preview)

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION AND CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Total lines read:        {stats['total_lines']}")
    print(f"Valid entries (prepared): {stats['valid_entries']}")
    print(f"Answer keys generated:   {stats['answer_keys_generated']}")
    total_skipped = sum(stats["skipped"].values())
    print(f"Skipped:                 {total_skipped}")
    if stats["skipped"]:
        print("\nSkip reasons:")
        for reason, count in sorted(stats["skipped"].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  Prepared: {prepared_output}")
    print(f"  Answer key: {answer_key_output}")


if __name__ == "__main__":
    main()
