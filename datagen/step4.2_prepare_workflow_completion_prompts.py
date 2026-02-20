import os
import json
import argparse
from tqdm import tqdm
from utils import (
    load_prompt_template,
    derive_answer_key_path,
    extract_raw_error,
    condense_trajectory,
    extract_final_response,
)

EVAL_SYSTEM_PROMPT = load_prompt_template('prompts/eval_workflow_completion.md')


def get_args():
    parser = argparse.ArgumentParser(description="Prepare Workflow Completion Evaluation Prompts")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the agent trajectory results file (*_results.jsonl).")
    parser.add_argument("--answer_key_file", type=str, default=None,
                        help="Path to the answer key / prepared file with metadata. If None, attempts to derive it.")
    return parser.parse_args()


def build_workflow_eval_prompt(question, cross_tool_workflow, tool_analysis,
                               condensed_trajectory, final_response):
    """
    Build the user-content portion of a workflow completion evaluation prompt.
    """
    parts = [
        f"## Original Question\n{question}",
        f"\n## Expected Cross-Tool Workflow\n{cross_tool_workflow}",
    ]

    if tool_analysis:
        parts.append(f"\n## Tool Analysis (Supporting Context)\n{tool_analysis}")

    parts.append(f"\n## Agent Trajectory (Condensed)\n{condensed_trajectory}")
    parts.append(f"\n## Agent's Final Response\n{final_response if final_response else '(No final response produced)'}")

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
        cross_tool_workflow = metadata.get('cross_tool_workflow', ak_meta.get('cross_tool_workflow', ''))
        tool_analysis = metadata.get('tool_analysis', ak_meta.get('tool_analysis', ''))

        # --- Check for missing workflow reference ---
        if not cross_tool_workflow:
            auto_scores.append({
                "metadata": {
                    "original_row_id": row_id,
                    "eval_dimension": "workflow_completion",
                },
                "score": None,
                "reasoning": "No cross_tool_workflow reference found in metadata.",
                "status": "missing_reference",
            })
            continue

        # --- Check for catastrophic failure ---
        is_failed, raw_error_string = extract_raw_error(messages)

        if is_failed:
            auto_scores.append({
                "metadata": {
                    "original_row_id": row_id,
                    "eval_dimension": "workflow_completion",
                },
                "score": 0,
                "reasoning": f"Scenario failed. Raw error: {raw_error_string}",
                "error_type": raw_error_string,
                "status": "scenario_failed",
            })
            continue

        # --- Build eval prompt (one per scenario) ---
        condensed = condense_trajectory(messages)
        final_response = extract_final_response(messages)

        user_content = build_workflow_eval_prompt(
            question=question,
            cross_tool_workflow=cross_tool_workflow,
            tool_analysis=tool_analysis,
            condensed_trajectory=condensed,
            final_response=final_response,
        )

        eval_prompts.append({
            "messages": [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "metadata": {
                "original_row_id": row_id,
                "eval_dimension": "workflow_completion",
            },
        })

    # --- Save outputs ---
    base_name = args.input_file.replace(".jsonl", "")

    eval_file = f"{base_name}_eval_workflow_completion_prepared.jsonl"
    with open(eval_file, 'w') as f:
        for item in eval_prompts:
            f.write(json.dumps(item) + "\n")

    auto_file = f"{base_name}_eval_workflow_completion_auto_scores.jsonl"
    with open(auto_file, 'w') as f:
        for item in auto_scores:
            f.write(json.dumps(item) + "\n")

    print(f"\nDone!")
    print(f"  - {len(eval_prompts)} eval prompts: {eval_file}")
    print(f"  - {len(auto_scores)} auto-scored entries: {auto_file}")


if __name__ == "__main__":
    main()
