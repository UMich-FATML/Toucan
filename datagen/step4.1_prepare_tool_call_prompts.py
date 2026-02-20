import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
from utils import (
    load_prompt_template,
    derive_answer_key_path,
    extract_raw_error,
)

EVAL_SYSTEM_PROMPT = load_prompt_template('prompts/evaluator.md')


def get_args():
    parser = argparse.ArgumentParser(description="Prepare Tool Call Evaluation Prompts for Agent Trajectories")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the agent trajectories (results) file.")
    parser.add_argument("--answer_key_file", type=str, default=None,
                        help="Path to the answer key file. If None, attempts to derive it.")
    return parser.parse_args()


def _parse_tool_args(raw_args):
    """Parse tool call arguments from string or dict."""
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except (json.JSONDecodeError, ValueError):
            return raw_args
    return raw_args


def extract_all_tool_calls(messages, tool_name):
    """
    Extract all calls to `tool_name` from the trajectory, chronologically.
    Returns list of argument dicts.
    """
    results = []
    for msg in messages:
        # OpenAI 'tool_calls' list
        if 'tool_calls' in msg and msg['tool_calls']:
            for tc in msg['tool_calls']:
                func = tc.get('function', {})
                if func.get('name') == tool_name:
                    results.append(_parse_tool_args(func.get('arguments', '{}')))

        # Legacy 'function_call' dict
        if 'function_call' in msg and msg['function_call']:
            func = msg['function_call']
            if func.get('name') == tool_name:
                results.append(_parse_tool_args(func.get('arguments', '{}')))

    return results


def build_single_eval_prompt(tool_name, expected_args, actual_args):
    """Build prompt for a single expected-vs-actual tool call comparison."""
    return (
        f"Please evaluate the following tool call pair.\n\n"
        f"EXPECTED Tool: {tool_name}\n"
        f"EXPECTED Arguments: {json.dumps(expected_args, indent=2)}\n\n"
        f"ACTUAL Tool: {tool_name}\n"
        f"ACTUAL Arguments: {json.dumps(actual_args, indent=2)}"
    )


def build_multi_eval_prompt(tool_name, expected_args_list, actual_args_list):
    """Build prompt for comparing multiple expected calls vs multiple actual calls."""
    parts = [
        f"Please evaluate the following MULTI-CALL tool comparison for tool: {tool_name}\n",
        f"The answer key expects {len(expected_args_list)} call(s) to this tool. "
        f"The agent made {len(actual_args_list)} call(s).\n",
    ]

    parts.append("## EXPECTED Calls (order does NOT matter):")
    for idx, args in enumerate(expected_args_list, 1):
        parts.append(f"\nExpected Call {idx}:\n{json.dumps(args, indent=2)}")

    parts.append("\n## ACTUAL Calls (order does NOT matter):")
    for idx, args in enumerate(actual_args_list, 1):
        parts.append(f"\nActual Call {idx}:\n{json.dumps(args, indent=2)}")

    return "\n".join(parts)


def main():
    args = get_args()

    answer_key_path = args.answer_key_file or derive_answer_key_path(args.input_file)
    if not answer_key_path or not os.path.exists(answer_key_path):
        print("Error: Answer key file not found.")
        print("  Tried to derive from input path. Pass --answer_key_file explicitly.")
        exit(1)

    print(f"Trajectories: {args.input_file}")
    print(f"Answer Key:   {answer_key_path}")

    with open(args.input_file, 'r') as f:
        trajectories = [json.loads(line) for line in f]

    with open(answer_key_path, 'r') as f:
        answer_keys_map = {}
        for i, line in enumerate(f):
            data = json.loads(line)
            rid = data.get('metadata', {}).get('row_id', i)
            answer_keys_map[rid] = data.get('answer_key', [])

    eval_prompts = []
    auto_scores = []

    print("Processing scenarios...")

    for i, run in enumerate(tqdm(trajectories)):
        row_id = run.get('metadata', {}).get('row_id', i)
        messages = run.get('messages', [])
        expected_chain = answer_keys_map.get(row_id)

        if not expected_chain:
            print(f"Warning: No answer key found for row {row_id}")
            continue

        # --- Check for global failure ---
        is_failed, raw_error_string = extract_raw_error(messages)

        # Group expected tools by name
        expected_by_name = defaultdict(list)
        for entry in expected_chain:
            name = entry.get('tool')
            expected_by_name[name].append(entry.get('arguments'))

        unique_tool_names = list(expected_by_name.keys())

        if is_failed:
            for tool_name in unique_tool_names:
                auto_scores.append({
                    "metadata": {
                        "original_row_id": row_id,
                        "eval_dimension": "tool_call",
                        "tool_name": tool_name,
                        "expected_count": len(expected_by_name[tool_name]),
                        "actual_count": 0,
                        "scenario_total_unique_tools": len(unique_tool_names),
                    },
                    "score": 0,
                    "reasoning": f"Scenario Failed. Raw Error: {raw_error_string}",
                    "error_type": raw_error_string,
                    "status": "scenario_failed"
                })
            continue

        # --- Process tools grouped by name ---
        for tool_name in unique_tool_names:
            expected_args_list = expected_by_name[tool_name]
            n_expected = len(expected_args_list)

            all_actual = extract_all_tool_calls(messages, tool_name)

            meta = {
                "original_row_id": row_id,
                "eval_dimension": "tool_call",
                "tool_name": tool_name,
                "expected_count": n_expected,
                "actual_count": len(all_actual),
                "scenario_total_unique_tools": len(unique_tool_names),
            }

            if not all_actual:
                # No calls to this tool at all
                auto_scores.append({
                    "metadata": meta,
                    "score": 0,
                    "reasoning": "Tool was not called by the agent.",
                    "error_type": "MISSING_TOOL_CALL",
                    "status": "missing_tool"
                })
            elif n_expected == 1:
                # Single expected call — use last actual
                user_content = build_single_eval_prompt(
                    tool_name, expected_args_list[0], all_actual[-1]
                )
                eval_prompts.append({
                    "messages": [
                        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    "metadata": meta
                })
            else:
                # Multiple expected calls — compare lists (take N most recent actual)
                actual_subset = all_actual[-n_expected:]
                user_content = build_multi_eval_prompt(
                    tool_name, expected_args_list, actual_subset
                )
                eval_prompts.append({
                    "messages": [
                        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    "metadata": meta
                })

    base_name = args.input_file.replace(".jsonl", "")

    # Save Prompts
    eval_file = f"{base_name}_eval_tool_call_prepared.jsonl"
    with open(eval_file, 'w') as f:
        for item in eval_prompts:
            f.write(json.dumps(item) + "\n")

    # Save Auto-Scores (Failures + Missing)
    auto_file = f"{base_name}_eval_tool_call_auto_scores.jsonl"
    with open(auto_file, 'w') as f:
        for item in auto_scores:
            f.write(json.dumps(item) + "\n")

    print(f"\nDone!")
    print(f"  - {len(eval_prompts)} eval prompts: {eval_file}")
    print(f"  - {len(auto_scores)} auto-scored failures: {auto_file}")

if __name__ == "__main__":
    main()
