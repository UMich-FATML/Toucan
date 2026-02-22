import argparse
import copy
import json
import os
import re
from time import sleep, time
from typing import Any, Dict, List

import requests
from tqdm import tqdm

try:
    from pydantic import BaseModel, ValidationError
except ImportError:
    print(
        "Error: pydantic is required for structured_completions_endpoint.py. "
        "Install it with: pip install pydantic"
    )
    raise SystemExit(1)

from utils import (
    get_model_abbreviation,
    load_dataset_from_file,
    safe_save_checkpoint,
    save_dataset,
)


class TargetToolModel(BaseModel):
    server: str
    tool: str
    arguments: Dict[str, Any]
    output: Any


class OnetResponseModel(BaseModel):
    tool_analysis: str
    cross_tool_workflow: str
    target_tools: List[TargetToolModel]
    request: str


RESPONSE_JSON_SCHEMA = {
    "name": "onet-task-response",
    "schema": OnetResponseModel.model_json_schema(),
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Structured O*NET completion endpoint with vLLM response_format."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Model path.")
    parser.add_argument("--input_file", type=str, required=True, help="Input prepared file.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10,
        help="Save checkpoint every n batches.",
    )
    parser.add_argument(
        "--vllm_api_url",
        type=str,
        default="http://localhost:8000/v1/chat/completions",
        help="vLLM API URL.",
    )
    parser.add_argument("--vllm_api_key", type=str, default="EMPTY", help="vLLM API key.")
    parser.add_argument(
        "--engine",
        type=str,
        default="vllm_api",
        choices=["vllm_api"],
        help="Only vllm_api is supported.",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="Repetition penalty."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32768,
        help="Max output tokens.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Retry count for malformed outputs.",
    )
    parser.add_argument(
        "--request_retries",
        type=int,
        default=3,
        help="HTTP retry count per request.",
    )
    parser.add_argument("--step", type=str, default="1.2_onet", help="Pipeline step tag.")
    return parser.parse_args()


args = get_args()
print(f"Structured O*NET Response Generation. Arguments: {args}")

if not args.input_file.endswith("prepared.jsonl") and not args.input_file.endswith("prepared.json"):
    raise ValueError("Input file must end with prepared.json(l) for completion pipeline.")

model_abbreviation = get_model_abbreviation(args.model_path)
base_name = args.input_file[: args.input_file.rfind(".")]
if base_name.endswith("_4prepared"):
    base_name = base_name[:-10]
elif base_name.endswith("_prepared"):
    base_name = base_name[:-9]
checkpoint_file = f"{base_name}_{model_abbreviation}_results_checkpoint.json"
saved_file = f"{base_name}_{model_abbreviation}_results.jsonl"

API_ENDPOINT = args.vllm_api_url
API_HEADERS = {
    "Authorization": f"Bearer {args.vllm_api_key}",
    "Content-Type": "application/json",
}


def request_completion(messages):
    payload = {
        "model": args.model_path,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": RESPONSE_JSON_SCHEMA,
        },
    }
    if args.max_tokens is not None and args.max_tokens > 0:
        payload["max_tokens"] = args.max_tokens

    for attempt in range(args.request_retries):
        try:
            response = requests.post(API_ENDPOINT, json=payload, headers=API_HEADERS)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, ValueError) as e:
            print(f"Request attempt {attempt + 1} failed: {e}")
            sleep(2**attempt)

    return ""


def parse_assistant_json(raw_text):
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None, "empty response"

    text = raw_text.strip()
    candidates = [text]

    codeblock_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if codeblock_match:
        candidates.append(codeblock_match.group(1).strip())

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            validated = OnetResponseModel.model_validate(parsed)
            return validated, ""
        except (json.JSONDecodeError, ValidationError):
            continue

    return None, "invalid JSON content or schema mismatch"


def build_repair_prompt(raw_output, parse_error):
    return (
        "Your previous response is invalid.\n"
        f"Issue: {parse_error}\n"
        "Rewrite the response to valid JSON only, matching this exact shape:\n"
        "{tool_analysis: string, cross_tool_workflow: string, target_tools: array, request: string}.\n"
        "Each target_tools item must contain: server (string), tool (string), arguments (object), output.\n"
        "Do not include markdown or explanations.\n"
        f"Previous response:\n{raw_output}"
    )


def process_item(item):
    original_messages = item.get("messages", [])
    if not original_messages:
        item["messages"] = [{"role": "assistant", "content": ""}]
        return item

    conversation = copy.deepcopy(original_messages)
    last_response = ""

    for attempt in range(args.max_retries + 1):
        response_text = request_completion(conversation).strip()
        parsed_model, parse_error = parse_assistant_json(response_text)

        if parsed_model is not None:
            normalized_json = json.dumps(parsed_model.model_dump(), ensure_ascii=False)
            item["messages"] = original_messages + [
                {"role": "assistant", "content": normalized_json}
            ]
            return item

        last_response = response_text
        if attempt < args.max_retries:
            repair_prompt = build_repair_prompt(response_text, parse_error)
            conversation = copy.deepcopy(original_messages)
            conversation.append({"role": "assistant", "content": response_text})
            conversation.append({"role": "user", "content": repair_prompt})

    item["messages"] = original_messages + [{"role": "assistant", "content": last_response}]
    return item


def add_generation_config_to_metadata(dataset):
    config_entry = {
        "model": model_abbreviation,
        "generation_params": {
            "engine": args.engine,
            "model_path": args.model_path,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "step": args.step,
        },
        "timestamp": int(time()),
    }

    for item in dataset:
        if "metadata" not in item:
            item["metadata"] = {}
        if "synthetic_data_gen_configs" not in item["metadata"]:
            item["metadata"]["synthetic_data_gen_configs"] = []
        item["metadata"]["synthetic_data_gen_configs"].append(config_entry)

    return dataset


def generate_and_update(dataset):
    processed_dataset = copy.deepcopy(dataset)

    if os.path.exists(checkpoint_file):
        checkpoint_dataset = load_dataset_from_file(checkpoint_file)
        last_checkpoint_idx = len(checkpoint_dataset)
        print(f"Checkpoint found. Resuming from index {last_checkpoint_idx}.")
        processed_dataset[:last_checkpoint_idx] = checkpoint_dataset
    else:
        last_checkpoint_idx = 0

    remaining = len(processed_dataset) - last_checkpoint_idx
    if remaining <= 0:
        print("No remaining items to process.")
    num_batches = (remaining + args.batch_size - 1) // args.batch_size if remaining > 0 else 0

    for i in tqdm(range(num_batches), desc="Generating completions"):
        start_idx = last_checkpoint_idx + i * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(processed_dataset))
        for idx in range(start_idx, end_idx):
            processed_dataset[idx] = process_item(processed_dataset[idx])

        if i % args.checkpoint_every == 0:
            safe_save_checkpoint(
                processed_dataset[:end_idx], checkpoint_file, convert_to_jsonl=False
            )
            print(f"Checkpoint saved after batch {i + 1}.")

    processed_dataset = add_generation_config_to_metadata(processed_dataset)
    return processed_dataset


def main():
    dataset = load_dataset_from_file(args.input_file)
    if not isinstance(dataset, list):
        dataset = [dataset]

    updated_dataset = generate_and_update(dataset)
    save_dataset(updated_dataset, saved_file, convert_to_jsonl=True)

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    print(f"Final dataset saved to {saved_file}.")


if __name__ == "__main__":
    main()
