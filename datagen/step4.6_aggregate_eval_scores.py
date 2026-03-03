#!/usr/bin/env python3
"""
Step 4.6: Aggregate Evaluation Scores Across All Dimensions

Merges per-dimension processed score files back to scenario level,
producing one record per original_row_id with all dimension scores.

Usage:
  python step4.6_aggregate_eval_scores.py \
    --tool_call_scores path/to/eval_tool_call_processed.jsonl \
    --workflow_completion_scores path/to/eval_workflow_completion_processed.jsonl \
    --grounding_scores path/to/eval_grounding_processed.jsonl \
    --output_file path/to/aggregated_scores.jsonl

  # With custom weights
  python step4.6_aggregate_eval_scores.py \
    --tool_call_scores ... --workflow_completion_scores ... --grounding_scores ... \
    --weight_tool_call 0.33 --weight_workflow_completion 0.34 --weight_grounding 0.33
"""

import os
import json
import argparse
from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser(description="Aggregate Evaluation Scores Across Dimensions")
    parser.add_argument("--tool_call_scores", type=str, default=None,
                        help="Processed tool-call scores JSONL (from step4.1 + step4.5 or existing pipeline).")
    parser.add_argument("--workflow_completion_scores", type=str, default=None,
                        help="Processed workflow completion scores JSONL.")
    parser.add_argument("--grounding_scores", type=str, default=None,
                        help="Processed grounding scores JSONL.")
    parser.add_argument("--followup_quality_scores", type=str, default=None,
                        help="Processed follow-up quality scores JSONL (withheld-info scenarios only). "
                             "When provided, included in the overall weighted mean.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output aggregated scores JSONL.")
    # Weights for overall score
    parser.add_argument("--weight_tool_call", type=float, default=1.0,
                        help="Weight for tool call accuracy in overall score.")
    parser.add_argument("--weight_workflow_completion", type=float, default=1.0,
                        help="Weight for workflow completion in overall score.")
    parser.add_argument("--weight_grounding", type=float, default=1.0,
                        help="Weight for grounding in overall score.")
    parser.add_argument("--weight_followup_quality", type=float, default=1.0,
                        help="Weight for follow-up quality in overall score (only applied when --followup_quality_scores is provided).")
    return parser.parse_args()


def load_scores(filepath):
    """Load a processed scores JSONL file. Returns list of dicts."""
    if not filepath or not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]


def aggregate_tool_call(entries):
    """
    Aggregate tool-call entries for a single scenario.
    Tool call scores are binary (0 or 1) per unique tool name.
    """
    scores = []
    for e in entries:
        sc = e.get("score")
        if sc is not None:
            scores.append(sc)
    if not scores:
        return None
    return {
        "scores": scores,
        "mean": sum(scores) / len(scores),
        "tools_evaluated": len(scores),
    }


def aggregate_workflow_completion(entries):
    """
    Aggregate workflow-completion entries for a single scenario.
    One entry per scenario, score 0-5.
    """
    if not entries:
        return None
    e = entries[0]
    sc = e.get("score")
    return {
        "score": sc if sc is not None else 0,
        "rating": e.get("rating", ""),
    }


def aggregate_grounding(entries):
    """
    Aggregate grounding entries for a single scenario.
    One entry per scenario, score 0-5.
    """
    if not entries:
        return None
    e = entries[0]
    sc = e.get("score")
    return {
        "score": sc if sc is not None else 0,
        "rating": e.get("rating", ""),
    }


def aggregate_followup_quality(entries):
    """
    Aggregate follow-up quality entries for a single scenario.
    One entry per scenario, score 1-5. Only present for withheld-info scenarios.
    Entries with status='missing_reference' are excluded (not withheld-info scenarios).
    """
    if not entries:
        return None
    # Skip missing_reference auto-scores — those are non-withheld scenarios
    scored = [e for e in entries if e.get("status") != "missing_reference" and e.get("score") is not None]
    if not scored:
        return None
    e = scored[0]
    sc = e.get("score")
    return {
        "score": sc,
        "rating": e.get("rating", ""),
    }


def main():
    args = get_args()

    # --- Load all dimension scores ---
    tool_call_data = load_scores(args.tool_call_scores)
    wf_data = load_scores(args.workflow_completion_scores)
    grounding_data = load_scores(args.grounding_scores)
    followup_data = load_scores(args.followup_quality_scores)

    print(f"Loaded scores:")
    print(f"  Tool call:            {len(tool_call_data)} entries")
    print(f"  Workflow completion:   {len(wf_data)} entries")
    print(f"  Grounding:            {len(grounding_data)} entries")
    if followup_data:
        print(f"  Follow-up quality:    {len(followup_data)} entries")

    # --- Group by original_row_id ---
    tc_by_row = defaultdict(list)
    for e in tool_call_data:
        rid = e.get("metadata", {}).get("original_row_id")
        if rid is not None:
            tc_by_row[rid].append(e)

    wf_by_row = defaultdict(list)
    for e in wf_data:
        rid = e.get("metadata", {}).get("original_row_id")
        if rid is not None:
            wf_by_row[rid].append(e)

    gr_by_row = defaultdict(list)
    for e in grounding_data:
        rid = e.get("metadata", {}).get("original_row_id")
        if rid is not None:
            gr_by_row[rid].append(e)

    fq_by_row = defaultdict(list)
    for e in followup_data:
        rid = e.get("metadata", {}).get("original_row_id")
        if rid is not None:
            fq_by_row[rid].append(e)

    # --- Collect all row IDs ---
    all_row_ids = sorted(
        set(tc_by_row.keys()) | set(wf_by_row.keys()) | set(gr_by_row.keys()) | set(fq_by_row.keys())
    )
    print(f"  Unique scenarios: {len(all_row_ids)}")

    # --- Aggregate ---
    results = []
    weights = {
        "tool_call": args.weight_tool_call,
        "workflow_completion": args.weight_workflow_completion,
        "grounding": args.weight_grounding,
        "followup_quality": args.weight_followup_quality,
    }

    for rid in all_row_ids:
        record = {"row_id": rid}

        # Tool call
        tc_agg = aggregate_tool_call(tc_by_row.get(rid, []))
        if tc_agg is not None:
            record["tool_call_accuracy"] = tc_agg

        # Workflow completion
        wf_agg = aggregate_workflow_completion(wf_by_row.get(rid, []))
        if wf_agg is not None:
            record["workflow_completion"] = wf_agg

        # Grounding
        gr_agg = aggregate_grounding(gr_by_row.get(rid, []))
        if gr_agg is not None:
            record["grounding"] = gr_agg

        # Follow-up quality (withheld-info scenarios only)
        fq_agg = aggregate_followup_quality(fq_by_row.get(rid, []))
        if fq_agg is not None:
            record["followup_quality"] = fq_agg

        # Overall weighted mean
        # Normalize scores to 0-1 range: tool_call is already 0-1, others are 1-5 → map to 0-1
        weighted_sum = 0.0
        active_weight = 0.0

        if tc_agg is not None:
            weighted_sum += weights["tool_call"] * tc_agg["mean"]  # already 0-1
            active_weight += weights["tool_call"]

        if wf_agg is not None:
            wf_score = wf_agg["score"] if wf_agg["score"] is not None else 0
            normalized = (wf_score - 1) / 4 if wf_score > 0 else 0
            weighted_sum += weights["workflow_completion"] * normalized
            active_weight += weights["workflow_completion"]

        if gr_agg is not None:
            gr_score = gr_agg["score"] if gr_agg["score"] is not None else 0
            normalized = (gr_score - 1) / 4 if gr_score > 0 else 0
            weighted_sum += weights["grounding"] * normalized
            active_weight += weights["grounding"]

        if fq_agg is not None:
            fq_score = fq_agg["score"] if fq_agg["score"] is not None else 0
            normalized = (fq_score - 1) / 4 if fq_score > 0 else 0
            weighted_sum += weights["followup_quality"] * normalized
            active_weight += weights["followup_quality"]

        if active_weight > 0:
            record["overall"] = round(weighted_sum / active_weight, 4)

        results.append(record)

    # --- Save ---
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    # --- Summary statistics ---
    overalls = [r["overall"] for r in results if "overall" in r]
    tc_means = [r["tool_call_accuracy"]["mean"] for r in results if "tool_call_accuracy" in r]
    wf_scores = [r["workflow_completion"]["score"] for r in results if "workflow_completion" in r and r["workflow_completion"]["score"] is not None]
    gr_scores = [r["grounding"]["score"] for r in results if "grounding" in r and r["grounding"]["score"] is not None]
    fq_scores = [r["followup_quality"]["score"] for r in results if "followup_quality" in r and r["followup_quality"]["score"] is not None]

    print(f"\n{'='*50}")
    print(f"Aggregation complete: {len(results)} scenarios")
    if tc_means:
        print(f"  Tool Call Accuracy     — mean: {sum(tc_means)/len(tc_means):.3f} (n={len(tc_means)})")
    if wf_scores:
        print(f"  Workflow Completion    — mean: {sum(wf_scores)/len(wf_scores):.2f}/5 (n={len(wf_scores)})")
    if gr_scores:
        print(f"  Grounding              — mean: {sum(gr_scores)/len(gr_scores):.2f}/5 (n={len(gr_scores)})")
    if fq_scores:
        print(f"  Follow-up Quality      — mean: {sum(fq_scores)/len(fq_scores):.2f}/5 (n={len(fq_scores)})")
    if overalls:
        print(f"  Overall (weighted)     — mean: {sum(overalls)/len(overalls):.3f} (n={len(overalls)})")
    print(f"Output: {args.output_file}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
