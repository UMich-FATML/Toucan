#!/usr/bin/env python3
"""
Step 4.5: Process LLM Judge Evaluation Scores

Parses JSON judge responses for all eval dimensions,
merges with auto-scores from the prepare step, and outputs
combined processed JSONL files.

Usage:
  # Process all dimensions at once
  python step4.5_process_eval_scores.py \
    --input_file path/to/results.jsonl \
    --dimensions all

  # Process a single dimension
  python step4.5_process_eval_scores.py \
    --input_file path/to/results.jsonl \
    --dimensions grounding

  # Process specific dimensions with custom output folder
  python step4.5_process_eval_scores.py \
    --input_file path/to/results.jsonl \
    --dimensions task_completion,grounding \
    --output_folder ../data/eval_processed
"""

import os
import glob
import json
import re
import argparse
from collections import Counter
from tqdm import tqdm

# Rating-to-score mappings per dimension
RATING_MAPS = {
    "workflow_completion": {
        "no workflow": 1,
        "wrong workflow": 2,
        "partial workflow": 3,
        "mostly complete": 4,
        "complete workflow": 5,
    },
    "grounding": {
        "ungrounded": 1,
        "poorly grounded": 2,
        "partially grounded": 3,
        "mostly grounded": 4,
        "fully grounded": 5,
    },
}

ALL_DIMENSIONS = ["tool_call", "workflow_completion", "grounding"]


def get_args():
    parser = argparse.ArgumentParser(description="Process LLM Judge Evaluation Scores")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Base results JSONL (the original agent trajectory file). "
                             "Results files are discovered by convention: {base}_eval_{dim}_*_results.jsonl")
    parser.add_argument("--dimensions", type=str, default="all",
                        help="Comma-separated dimensions to process, or 'all'. Default: all")
    parser.add_argument("--output_folder", type=str, default=None,
                        help="Output folder. Defaults to same directory as input.")
    return parser.parse_args()


def find_results_file(base_path, dimension):
    """
    Find the results file for a dimension by globbing.
    Pattern: {base}_eval_{dim}_*_results.jsonl
    """
    pattern = f"{base_path}_eval_{dimension}_*_results.jsonl"
    matches = glob.glob(pattern)
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Pick most recent
        matches.sort(key=os.path.getmtime, reverse=True)
        print(f"  Warning: Multiple results files found for '{dimension}', using most recent: {os.path.basename(matches[0])}")
        return matches[0]
    return None


def find_auto_scores_file(base_path, dimension):
    """
    Find the auto-scores file for a dimension.
    Pattern: {base}_eval_{dim}_auto_scores.jsonl
    """
    path = f"{base_path}_eval_{dimension}_auto_scores.jsonl"
    if os.path.exists(path):
        return path
    return None


def parse_judge_response(content, dimension):
    """
    Parse a JSON judge response string and extract score + metadata.
    Returns a dict with parsed fields, or None on failure.
    """
    if not content:
        return None

    content = content.strip()

    # Try to extract JSON from markdown code blocks first
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find a raw JSON object
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return None

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    # Extract score — prefer explicit numeric score, fall back to rating text
    score = parsed.get("score")
    rating = parsed.get("rating", "")

    if score is not None:
        try:
            score = int(score)
            # tool_call is binary 0/1; task_completion and grounding are 1-5
            if dimension == "tool_call":
                if score not in (0, 1):
                    score = None
            else:
                if not (1 <= score <= 5):
                    score = None
        except (ValueError, TypeError):
            score = None

    # If no valid numeric score, map from rating text
    if score is None and rating and dimension in RATING_MAPS:
        rating_lower = rating.strip().lower()
        score = RATING_MAPS[dimension].get(rating_lower)
        # Try partial match
        if score is None:
            for key, val in RATING_MAPS[dimension].items():
                if key in rating_lower or rating_lower in key:
                    score = val
                    break

    result = {
        "score": score,
        "rating": rating,
        "reasoning": parsed.get("reasoning", ""),
    }

    # Dimension-specific fields
    if dimension == "task_completion":
        result["evidence"] = parsed.get("evidence", "")
    elif dimension == "grounding":
        result["grounded_claims"] = parsed.get("grounded_claims", [])
        result["ungrounded_claims"] = parsed.get("ungrounded_claims", [])

    return result


def process_dimension(results_file, auto_scores_file, dimension, output_dir):
    """Process a single dimension's results. Returns summary dict."""
    print(f"\n{'─'*50}")
    print(f"Dimension: {dimension}")
    print(f"  Results:     {os.path.basename(results_file)}")

    # --- Load judge results ---
    with open(results_file, 'r') as f:
        judge_results = [json.loads(line) for line in f]
    print(f"  Loaded {len(judge_results)} judge entries")

    # --- Load auto-scores ---
    auto_scores = []
    if auto_scores_file and os.path.exists(auto_scores_file):
        print(f"  Auto-scores: {os.path.basename(auto_scores_file)}")
        with open(auto_scores_file, 'r') as f:
            auto_scores = [json.loads(line) for line in f]
        print(f"  Loaded {len(auto_scores)} auto-scored entries")
    else:
        print("  No auto-scores file found.")

    # --- Process judge results ---
    processed = []
    parse_failures = 0
    score_dist = Counter()

    for entry in tqdm(judge_results, desc=f"Processing {dimension}"):
        meta = entry.get("metadata", {})

        entry_dim = dimension or meta.get("eval_dimension", "unknown")

        # The judge response is in the completion output
        response_content = entry.get("response", "")
        if not response_content:
            msgs = entry.get("messages", [])
            if msgs and msgs[-1].get("role") == "assistant":
                response_content = msgs[-1].get("content", "")

        parsed = parse_judge_response(response_content, entry_dim)

        if parsed and parsed["score"] is not None:
            record = {
                "metadata": meta,
                "score": parsed["score"],
                "rating": parsed["rating"],
                "reasoning": parsed["reasoning"],
                "status": "judge_scored",
            }
            # Copy dimension-specific fields
            if "evidence" in parsed:
                record["evidence"] = parsed["evidence"]
            if "grounded_claims" in parsed:
                record["grounded_claims"] = parsed["grounded_claims"]
                record["ungrounded_claims"] = parsed["ungrounded_claims"]

            processed.append(record)
            score_dist[parsed["score"]] += 1
        else:
            parse_failures += 1
            processed.append({
                "metadata": meta,
                "score": None,
                "reasoning": "Failed to parse judge response",
                "raw_response": response_content[:500] if response_content else "",
                "status": "parse_failure",
            })

    # --- Merge auto-scores ---
    for entry in auto_scores:
        processed.append(entry)
        sc = entry.get("score")
        if sc is not None:
            score_dist[sc] += 1

    # --- Save output ---
    os.makedirs(output_dir, exist_ok=True)
    input_basename = os.path.basename(results_file)
    output_basename = input_basename.replace("_results.jsonl", "_processed.jsonl")
    output_path = os.path.join(output_dir, output_basename)

    with open(output_path, 'w') as f:
        for item in processed:
            f.write(json.dumps(item) + "\n")

    # --- Summary ---
    total = len(processed)
    judge_scored = sum(1 for p in processed if p.get("status") == "judge_scored")
    auto_scored = sum(1 for p in processed if p.get("status") in ("scenario_failed", "missing_tool", "no_content"))
    valid_scores = [p["score"] for p in processed if p.get("score") is not None]

    print(f"  Total entries: {total}")
    print(f"    Judge-scored: {judge_scored}")
    print(f"    Auto-scored:  {auto_scored}")
    print(f"    Parse failures: {parse_failures}")
    if valid_scores:
        print(f"  Score distribution:")
        for score in sorted(score_dist.keys()):
            count = score_dist[score]
            pct = count / len(valid_scores) * 100
            print(f"    {score}: {count} ({pct:.1f}%)")
        print(f"  Mean score: {sum(valid_scores) / len(valid_scores):.2f}")
    print(f"  Output: {output_path}")

    return {
        "dimension": dimension,
        "total": total,
        "judge_scored": judge_scored,
        "auto_scored": auto_scored,
        "parse_failures": parse_failures,
        "mean": sum(valid_scores) / len(valid_scores) if valid_scores else None,
        "output_path": output_path,
    }


def main():
    args = get_args()

    base_path = args.input_file.replace(".jsonl", "")

    # Expand dimensions
    if args.dimensions == "all":
        dimensions = ALL_DIMENSIONS
    else:
        dimensions = [d.strip() for d in args.dimensions.split(",")]

    output_dir = args.output_folder or os.path.dirname(args.input_file) or "."

    print(f"[Step 4.5] Process Evaluation Scores")
    print(f"  Input file:  {args.input_file}")
    print(f"  Dimensions:  {', '.join(dimensions)}")
    print(f"  Output dir:  {output_dir}")

    summaries = []
    skipped = []

    for dim in dimensions:
        results_file = find_results_file(base_path, dim)
        if not results_file:
            print(f"\n  Skipping '{dim}': no results file found matching {base_path}_eval_{dim}_*_results.jsonl")
            skipped.append(dim)
            continue

        auto_scores_file = find_auto_scores_file(base_path, dim)
        summary = process_dimension(results_file, auto_scores_file, dim, output_dir)
        summaries.append(summary)

    # --- Final summary ---
    print(f"\n{'='*50}")
    print(f"[Step 4.5] Complete")
    print(f"  Processed: {len(summaries)} dimension(s)")
    if skipped:
        print(f"  Skipped:   {', '.join(skipped)}")
    for s in summaries:
        mean_str = f"{s['mean']:.2f}" if s['mean'] is not None else "N/A"
        print(f"  {s['dimension']:20s} — {s['total']} entries, mean: {mean_str}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
