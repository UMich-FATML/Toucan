"""
Merge quality check results from server_quality_results_merged2.jsonl into
the individual server JSON files in mcp_servers/.

Adds a 'quality_check' field to each server JSON in-place.

Example usage:
    python step0_filter_mcp_servers/merge_quality_into_servers.py \
        --quality_results_file ../data/server_quality_results_merged2.jsonl \
        --server_dir ../mcp_servers/smithery_mcp_servers_0210/
"""

import argparse
import json
import os
import sys

# Allow imports from datagen/ (parent directory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from utils import load_dataset_from_file


def get_args():
    parser = argparse.ArgumentParser(
        description="Merge quality check results into server JSON files."
    )
    parser.add_argument(
        "--quality_results_file",
        type=str,
        required=True,
        help="Path to server_quality_results_merged*.jsonl.",
    )
    parser.add_argument(
        "--server_dir",
        type=str,
        required=True,
        help="Directory containing server JSON files to update in-place.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    print(f"Loading quality results from {args.quality_results_file}...")
    results = load_dataset_from_file(args.quality_results_file)
    if not isinstance(results, list):
        results = [results]
    print(f"Loaded {len(results)} server quality entries.")

    updated = 0
    skipped = 0

    for entry in tqdm(results, desc="Merging into server JSONs"):
        server_id = entry.get("server_id")
        if not server_id:
            skipped += 1
            continue

        server_path = os.path.join(args.server_dir, f"{server_id}.json")
        if not os.path.exists(server_path):
            print(f"  Warning: {server_path} not found, skipping.")
            skipped += 1
            continue

        with open(server_path, "r", encoding="utf-8") as f:
            server_data = json.load(f)

        meta = entry.get("metadata", {})
        server_data["quality_check"] = {
            "tool_results": entry.get("tool_results", []),
            "agent_error": entry.get("agent_error", False),
            "agent_error_type": entry.get("agent_error_type"),
            "num_tools_checked": meta.get("num_tools_checked", 0),
            "num_tools_passed": meta.get("num_tools_passed", 0),
            "model": meta.get("model", "unknown"),
            "timestamp": meta.get("timestamp"),
        }

        with open(server_path, "w", encoding="utf-8") as f:
            json.dump(server_data, f, ensure_ascii=False, indent=2)

        updated += 1

    print(f"\nDone. Updated {updated} server JSON files, skipped {skipped}.")


if __name__ == "__main__":
    main()
