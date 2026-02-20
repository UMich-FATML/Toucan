import os
import json
import argparse
from tqdm import tqdm
from utils import (
    load_prompt_template,
    derive_answer_key_path,
    extract_raw_error,
    extract_tool_evidence,
    extract_assistant_content,
)

EVAL_SYSTEM_PROMPT = load_prompt_template('prompts/eval_grounding.md')

def get_args():
    parser = argparse.ArgumentParser(description="Prepare Grounding Evaluation Prompts")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the agent trajectory results file (*_results.jsonl).")
    parser.add_argument("--answer_key_file", type=str, default=None,
                        help="Path to the answer key / prepared file with metadata. If None, attempts to derive it.")
    return parser.parse_args()


def format_tool_evidence(evidence_list):
    """Format extracted tool evidence into readable text for the judge."""
    if not evidence_list:
        return "(No tool calls were made)"

    parts = []
    for idx, ev in enumerate(evidence_list, 1):
        parts.append(f"### Tool Call {idx}: {ev['tool_name']}")
        args_display = ev['arguments_summary']
        if len(args_display) > 500:
            args_display = args_display[:500] + " ... (truncated)"
        parts.append(f"**Arguments**: {args_display}")

        output_display = ev['output'] if ev['output'] else "(no output received)"
        if len(output_display) > 1500:
            output_display = output_display[:1500] + " ... (truncated)"
        parts.append(f"**Output**: {output_display}")
        parts.append("")  # blank line separator

    return "\n".join(parts)


def build_grounding_eval_prompt(question, tool_evidence_text, assistant_content):
    """Build the user-content portion of a grounding evaluation prompt."""
    parts = [
        f"## Original Question\n{question}",
        f"\n## Tool Calls and Outputs (Evidence Base)\n{tool_evidence_text}",
        f"\n## Assistant Messages (Claims to Evaluate)\n{assistant_content if assistant_content else '(No assistant messages produced)'}",
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

        # --- Check for catastrophic failure ---
        is_failed, raw_error_string = extract_raw_error(messages)

        if is_failed:
            auto_scores.append({
                "metadata": {
                    "original_row_id": row_id,
                    "eval_dimension": "grounding",
                },
                "score": 0,
                "reasoning": f"Scenario failed. Raw error: {raw_error_string}",
                "error_type": raw_error_string,
                "status": "scenario_failed",
            })
            continue

        # --- Check for empty assistant content ---
        assistant_content = extract_assistant_content(messages)

        if not assistant_content.strip():
            auto_scores.append({
                "metadata": {
                    "original_row_id": row_id,
                    "eval_dimension": "grounding",
                },
                "score": 0,
                "reasoning": "Agent produced no assistant text content.",
                "error_type": "NO_ASSISTANT_CONTENT",
                "status": "no_content",
            })
            continue

        # --- Build eval prompt ---
        tool_evidence = extract_tool_evidence(messages)
        tool_evidence_text = format_tool_evidence(tool_evidence)

        user_content = build_grounding_eval_prompt(
            question=question,
            tool_evidence_text=tool_evidence_text,
            assistant_content=assistant_content,
        )

        eval_prompts.append({
            "messages": [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "metadata": {
                "original_row_id": row_id,
                "eval_dimension": "grounding",
            },
        })

    # --- Save outputs ---
    base_name = args.input_file.replace(".jsonl", "")

    eval_file = f"{base_name}_eval_grounding_prepared.jsonl"
    with open(eval_file, 'w') as f:
        for item in eval_prompts:
            f.write(json.dumps(item) + "\n")

    auto_file = f"{base_name}_eval_grounding_auto_scores.jsonl"
    with open(auto_file, 'w') as f:
        for item in auto_scores:
            f.write(json.dumps(item) + "\n")

    print(f"\nDone!")
    print(f"  - {len(eval_prompts)} eval prompts: {eval_file}")
    print(f"  - {len(auto_scores)} auto-scored failures: {auto_file}")


if __name__ == "__main__":
    main()
