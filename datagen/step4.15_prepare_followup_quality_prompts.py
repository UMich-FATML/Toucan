#!/usr/bin/env python3
"""
Step 4.15: Prepare Follow-up Quality Evaluation Prompts

Evaluates whether an agent correctly identified and asked for missing information
before making tool calls. Only generates eval prompts for scenarios that have
withheld_info in their metadata; other scenarios are auto-scored as missing_reference.

Usage:
  python step4.15_prepare_followup_quality_prompts.py \
    --input_file path/to/results.jsonl

  python step4.15_prepare_followup_quality_prompts.py \
    --input_file path/to/results.jsonl \
    --answer_key_file path/to/answer_key.jsonl
"""

import os
import json
import argparse
from tqdm import tqdm
from utils import (
    load_prompt_template,
    derive_answer_key_path,
    extract_raw_error,
    condense_trajectory,
)

EVAL_SYSTEM_PROMPT = load_prompt_template('prompts/eval_followup_quality.md')


def get_args():
    parser = argparse.ArgumentParser(description="Prepare Follow-up Quality Evaluation Prompts")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the agent trajectory results file (*_results.jsonl).")
    parser.add_argument("--answer_key_file", type=str, default=None,
                        help="Path to the answer key / prepared file with metadata. If None, attempts to derive it.")
    return parser.parse_args()


def format_withheld_info(withheld_info: list) -> str:
    """Format withheld_info list for the eval prompt."""
    lines = []
    for item in withheld_info:
        param = item.get("parameter", "")
        desc = item.get("description", "")
        value = item.get("value", "")
        lines.append(f"- **{param}**: {desc} → correct value: `{value}`")
    return "\n".join(lines)


def format_target_followup_questions(questions: list) -> str:
    """Format target follow-up questions for the eval prompt."""
    return "\n".join(f"- {q}" for q in questions)


def build_followup_eval_prompt(question, withheld_info, target_followup_questions,
                                condensed_trajectory):
    """
    Build the user-content portion of a follow-up quality evaluation prompt.
    """
    parts = [
        f"## Original Request (with deliberate omissions)\n{question}",
        f"\n## Withheld Information (parameters the agent needed to ask about)\n{format_withheld_info(withheld_info)}",
        f"\n## Target Follow-up Questions (what the agent should have asked)\n{format_target_followup_questions(target_followup_questions)}",
        f"\n## Agent Trajectory (Condensed)\n{condensed_trajectory}",
    ]
    return "\n".join(parts)


def main():
    args = get_args()

    # --- Load trajectories ---
    print(f"Loading trajectories from: {args.input_file}")
    with open(args.input_file, 'r') as f:
        trajectories = [json.loads(line) for line in f]
    print(f"  Loaded {len(trajectories)} scenarios")

    # --- Optionally load answer key for metadata ---
    answer_key_path = args.answer_key_file or derive_answer_key_path(args.input_file)
    answer_keys_map = {}
    if answer_key_path and os.path.exists(answer_key_path):
        print(f"Loading answer key from: {answer_key_path}")
        with open(answer_key_path, 'r') as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                rid = data.get('metadata', {}).get('row_id', i)
                answer_keys_map[rid] = data
    else:
        print("No answer key file found — will rely on trajectory metadata only.")

    # --- Process scenarios ---
    eval_prompts = []
    auto_scores = []

    print("Processing scenarios...")
    for i, run in enumerate(tqdm(trajectories)):
        metadata = run.get('metadata', {})
        row_id = metadata.get('row_id', i)
        messages = run.get('messages', [])

        ak_meta = answer_keys_map.get(row_id, {}).get('metadata', {})

        question = metadata.get('question', ak_meta.get('question', ''))
        withheld_info = metadata.get('withheld_info', ak_meta.get('withheld_info', None))
        target_followup_questions = metadata.get(
            'target_followup_questions',
            ak_meta.get('target_followup_questions', None)
        )

        # --- Auto-score scenarios without withheld_info (not withheld-info scenarios) ---
        if not withheld_info or not isinstance(withheld_info, list) or len(withheld_info) == 0:
            auto_scores.append({
                "metadata": {
                    "original_row_id": row_id,
                    "eval_dimension": "followup_quality",
                },
                "score": None,
                "reasoning": "No withheld_info found in metadata — not a withheld-info scenario.",
                "status": "missing_reference",
            })
            continue

        # Ensure we have at least some target questions (use empty list if missing)
        if not target_followup_questions or not isinstance(target_followup_questions, list):
            target_followup_questions = []

        # --- Check for catastrophic failure ---
        is_failed, raw_error_string = extract_raw_error(messages)
        if is_failed:
            auto_scores.append({
                "metadata": {
                    "original_row_id": row_id,
                    "eval_dimension": "followup_quality",
                },
                "score": 0,
                "reasoning": f"Scenario failed. Raw error: {raw_error_string}",
                "error_type": raw_error_string,
                "status": "scenario_failed",
            })
            continue

        # --- Build eval prompt ---
        condensed = condense_trajectory(messages)

        user_content = build_followup_eval_prompt(
            question=question,
            withheld_info=withheld_info,
            target_followup_questions=target_followup_questions,
            condensed_trajectory=condensed,
        )

        eval_prompts.append({
            "messages": [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "metadata": {
                "original_row_id": row_id,
                "eval_dimension": "followup_quality",
            },
        })

    # --- Save outputs ---
    base_name = args.input_file.replace(".jsonl", "")

    eval_file = f"{base_name}_eval_followup_quality_prepared.jsonl"
    with open(eval_file, 'w') as f:
        for item in eval_prompts:
            f.write(json.dumps(item) + "\n")

    auto_file = f"{base_name}_eval_followup_quality_auto_scores.jsonl"
    with open(auto_file, 'w') as f:
        for item in auto_scores:
            f.write(json.dumps(item) + "\n")

    print(f"\nDone!")
    print(f"  - {len(eval_prompts)} eval prompts (withheld-info scenarios): {eval_file}")
    print(f"  - {len(auto_scores)} auto-scored entries (non-withheld or failed): {auto_file}")


if __name__ == "__main__":
    main()
