import asyncio
import argparse
import copy
import json
import os
import re
import shutil
from time import time

from openai import AsyncOpenAI
from tqdm import tqdm

try:
  from jsonschema import Draft202012Validator, ValidationError as JsonSchemaValidationError
except ImportError:
  print(
    "Error: jsonschema is required for structured_completions_endpoint.py. "
    "Install it with: pip install jsonschema"
  )
  raise SystemExit(1)

from utils import (
  get_model_abbreviation,
  load_dataset_from_file,
  save_dataset,
)


def get_default_output_schema_path():
  script_dir = os.path.dirname(os.path.abspath(__file__))
  return os.path.join(script_dir, "prompts", "genq_from_onet_tasks_output_schema.json")


def derive_schema_name(schema_path):
  base_name = os.path.splitext(os.path.basename(schema_path))[0]
  normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", base_name).strip("-")
  return normalized or "structured-response"


def load_response_schema_config(schema_path):
  if not os.path.exists(schema_path):
    raise FileNotFoundError(f"Output schema file not found: {schema_path}")

  with open(schema_path, "r", encoding="utf-8") as f:
    schema_data = json.load(f)

  if not isinstance(schema_data, dict):
    raise ValueError("Output schema must be a JSON object.")

  if isinstance(schema_data.get("schema"), dict):
    normalized_schema = schema_data["schema"]
    normalized_name = schema_data.get("name", derive_schema_name(schema_path))
  else:
    normalized_schema = schema_data
    normalized_name = derive_schema_name(schema_path)

  Draft202012Validator.check_schema(normalized_schema)
  return (
    {
      "name": normalized_name,
      "schema": normalized_schema,
    },
    Draft202012Validator(normalized_schema),
  )


def get_args():
  parser = argparse.ArgumentParser(
    description="Structured O*NET completion endpoint with vLLM response_format."
  )
  parser.add_argument("--model_name", type=str, required=True, help="Model name.")
  parser.add_argument("--input_file", type=str, required=True, help="Input prepared file.")
  parser.add_argument(
    "--start_idx",
    type=int,
    default=0,
    help="Start index (inclusive) of rows to process.",
  )
  parser.add_argument(
    "--batch_size",
    type=int,
    default=None,
    help="Optional number of rows to process from start_idx.",
  )
  parser.add_argument("--concurrency", type=int, default=100, help="Max concurrent requests.")
  parser.add_argument(
    "--base_url",
    type=str,
    default="http://localhost:8000/v1",
    help="vLLM API base URL, e.g. http://localhost:8000/v1.",
  )
  parser.add_argument("--api_key", type=str, default="EMPTY", help="vLLM API key.")
  parser.add_argument(
    "--engine",
    type=str,
    default="vllm_api",
    choices=["vllm_api", "openrouter_api"],
    help="API engine to use.",
  )
  parser.add_argument(
    "--openrouter_url",
    type=str,
    default="https://openrouter.ai/api/v1",
    help="OpenRouter API URL.",
  )
  parser.add_argument(
    "--openrouter_api_key",
    type=str,
    default="",
    help="OpenRouter API Key (or set OPENROUTER_API_KEY env var).",
  )
  parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
  parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling.")
  parser.add_argument(
    "--repetition_penalty", type=float, default=1.0, help="Repetition penalty."
  )
  parser.add_argument(
    "--max_tokens",
    type=int,
    default=40960,
    help="Max output tokens.",
  )
  parser.add_argument(
    "--api_retries",
    type=int,
    default=3,
    help="Retry count per API request.",
  )
  parser.add_argument(
    "--output_schema_file",
    type=str,
    default=get_default_output_schema_path(),
    help="Path to JSON schema file used for response_format and validation.",
  )
  parser.add_argument("--step", type=str, default="1.2_onet", help="Pipeline step tag.")
  return parser.parse_args()


args = get_args()
print(f"Structured O*NET Response Generation. Arguments: {args}")

if not args.input_file.endswith("prepared.jsonl") and not args.input_file.endswith("prepared.json"):
  raise ValueError("Input file must end with prepared.json(l) for completion pipeline.")
if args.concurrency <= 0:
  raise ValueError("--concurrency must be a positive integer.")
if args.start_idx < 0:
  raise ValueError("--start_idx must be a non-negative integer.")
if args.batch_size is not None and args.batch_size <= 0:
  raise ValueError("--batch_size must be a positive integer.")

if args.engine == "openrouter_api":
  _resolved_api_key = args.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
  if not _resolved_api_key:
    raise ValueError(
      "OpenRouter API Key is required when using openrouter_api engine. "
      "Provide --openrouter_api_key or set OPENROUTER_API_KEY env var."
    )
  _resolved_base_url = args.openrouter_url.rstrip("/")
else:
  _base_url = args.base_url.rstrip("/")
  if not _base_url.endswith("/v1"):
    raise ValueError("--base_url must end with /v1.")
  _resolved_base_url = _base_url
  _resolved_api_key = args.api_key

try:
  RESPONSE_JSON_SCHEMA, OUTPUT_SCHEMA_VALIDATOR = load_response_schema_config(
    args.output_schema_file
  )
except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
  raise SystemExit(f"Failed to load output schema: {e}")

print(f"Loaded output schema from: {args.output_schema_file}")

model_abbreviation = get_model_abbreviation(args.model_name)


def extract_message_text(content):
  if isinstance(content, str):
    return content
  if content is None:
    return ""
  if isinstance(content, list):
    text_chunks = []
    for part in content:
      if isinstance(part, dict) and part.get("type") == "text":
        text_chunks.append(part.get("text", ""))
    return "".join(text_chunks)
  return str(content)


async def request_completion_async(messages, client):
  payload = {
    "model": args.model_name,
    "temperature": args.temperature,
    "top_p": args.top_p,
    "messages": messages,
    "response_format": {
      "type": "json_schema",
      "json_schema": RESPONSE_JSON_SCHEMA,
    },
  }
  if args.max_tokens is not None and args.max_tokens > 0:
    payload["max_tokens"] = args.max_tokens
  if args.engine == "vllm_api":
    payload["extra_body"] = {"repetition_penalty": args.repetition_penalty}

  for attempt in range(args.api_retries):
    try:
      completion = await client.chat.completions.create(**payload)
      return extract_message_text(completion.choices[0].message.content)
    except Exception as e:
      print(f"Request attempt {attempt + 1} failed: {e}")
      await asyncio.sleep(2**attempt)

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

  last_schema_error = "invalid JSON content"
  for candidate in candidates:
    try:
      parsed = json.loads(candidate)
    except json.JSONDecodeError:
      continue
    try:
      OUTPUT_SCHEMA_VALIDATOR.validate(parsed)
      return parsed, ""
    except JsonSchemaValidationError as e:
      last_schema_error = e.message

  return None, f"schema mismatch: {last_schema_error}"


async def process_item_async(item, client):
  processed_item = copy.deepcopy(item)
  original_messages = processed_item.get("messages", [])
  if not original_messages:
    processed_item["messages"] = [{"role": "assistant", "content": ""}]
    return processed_item

  response_text = (await request_completion_async(original_messages, client)).strip()
  parsed_model, _ = parse_assistant_json(response_text)

  if parsed_model is not None:
    normalized_json = json.dumps(parsed_model, ensure_ascii=False)
    processed_item["messages"] = original_messages + [
      {"role": "assistant", "content": normalized_json}
    ]
    return processed_item

  processed_item["messages"] = original_messages + [{"role": "assistant", "content": response_text}]
  return processed_item


def checkpoint_file_path(index, checkpoint_dir):
  return os.path.join(checkpoint_dir, f"{index:08d}.json")


def save_item_checkpoint(index, item, checkpoint_dir):
  os.makedirs(checkpoint_dir, exist_ok=True)
  checkpoint_path = checkpoint_file_path(index, checkpoint_dir)
  temp_path = f"{checkpoint_path}.tmp"
  with open(temp_path, "w", encoding="utf-8") as f:
    json.dump(item, f, ensure_ascii=False)
    f.write("\n")
  os.replace(temp_path, checkpoint_path)


def load_item_checkpoints(processed_dataset, checkpoint_dir):
  completed_indices = set()
  if not os.path.isdir(checkpoint_dir):
    return completed_indices

  for file_name in os.listdir(checkpoint_dir):
    match = re.fullmatch(r"(\d+)\.json", file_name)
    if match is None:
      continue

    index = int(match.group(1))
    if index < 0 or index >= len(processed_dataset):
      continue

    file_path = os.path.join(checkpoint_dir, file_name)
    try:
      with open(file_path, "r", encoding="utf-8") as f:
        processed_dataset[index] = json.load(f)
      completed_indices.add(index)
    except (OSError, json.JSONDecodeError) as e:
      print(f"Failed to load checkpoint {file_path}: {e}. Reprocessing index {index}.")

  return completed_indices


def add_generation_config_to_metadata(dataset):
  config_entry = {
    "model": model_abbreviation,
    "generation_params": {
      "engine": args.engine,
      "model_name": args.model_name,
      "concurrency": args.concurrency,
      "temperature": args.temperature,
      "max_tokens": args.max_tokens,
      "top_p": args.top_p,
      "repetition_penalty": args.repetition_penalty,
      "step": args.step,
      "output_schema_file": args.output_schema_file,
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


async def generate_and_update(dataset, client, checkpoint_dir):
  processed_dataset = copy.deepcopy(dataset)
  os.makedirs(checkpoint_dir, exist_ok=True)

  completed_indices = load_item_checkpoints(processed_dataset, checkpoint_dir)
  if completed_indices:
    print(
      f"Loaded {len(completed_indices)} completed item checkpoints from {checkpoint_dir}."
    )

  pending_indices = [idx for idx in range(len(processed_dataset)) if idx not in completed_indices]
  if not pending_indices:
    print("No remaining items to process.")
  else:
    print(
      f"Processing {len(pending_indices)} items with max concurrency {args.concurrency}."
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def process_index(index):
      async with semaphore:
        try:
          processed_item = await process_item_async(processed_dataset[index], client)
          return index, processed_item
        except Exception as e:
          print(f"Failed to process index {index}: {e}")
          fallback_item = copy.deepcopy(processed_dataset[index])
          original_messages = fallback_item.get("messages", [])
          fallback_item["messages"] = original_messages + [
            {"role": "assistant", "content": ""}
          ]
          return index, fallback_item

    tasks = [asyncio.create_task(process_index(idx)) for idx in pending_indices]
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating completions"):
      index, processed_item = await task
      processed_dataset[index] = processed_item
      save_item_checkpoint(index, processed_item, checkpoint_dir)

  processed_dataset = add_generation_config_to_metadata(processed_dataset)
  return processed_dataset


def get_input_base_name(input_file):
  base_name = input_file[: input_file.rfind(".")]
  if base_name.endswith("_4prepared"):
    return base_name[:-10]
  if base_name.endswith("_prepared"):
    return base_name[:-9]
  return base_name


def resolve_processing_range(total_rows):
  if total_rows <= 0:
    raise ValueError(f"Input dataset is empty: {args.input_file}")
  if args.start_idx >= total_rows:
    raise ValueError(
      f"--start_idx ({args.start_idx}) must be smaller than dataset size ({total_rows})."
    )

  start_idx = args.start_idx
  requested_end_idx = (
    total_rows if args.batch_size is None else args.start_idx + args.batch_size
  )
  end_idx = min(requested_end_idx, total_rows)
  return start_idx, requested_end_idx, end_idx


def build_output_paths(base_name, start_idx, end_idx):
  range_mode = args.batch_size is not None or args.start_idx != 0
  if range_mode:
    saved_file = f"{base_name}_{model_abbreviation}_results_{start_idx}_{end_idx}.jsonl"
    checkpoint_dir = (
      f"{base_name}_{model_abbreviation}_results_checkpoints_{start_idx}_{end_idx}"
    )
  else:
    saved_file = f"{base_name}_{model_abbreviation}_results.jsonl"
    checkpoint_dir = f"{base_name}_{model_abbreviation}_results_checkpoints"
  return saved_file, checkpoint_dir


async def main():
  dataset = load_dataset_from_file(args.input_file)
  if not isinstance(dataset, list):
    dataset = [dataset]
  total_rows = len(dataset)
  start_idx, requested_end_idx, end_idx = resolve_processing_range(total_rows)
  target_dataset = dataset[start_idx:end_idx]

  base_name = get_input_base_name(args.input_file)
  saved_file, checkpoint_dir = build_output_paths(base_name, start_idx, end_idx)

  print(
    f"Dataset rows: {total_rows}. Requested range: "
    f"[{start_idx}, {requested_end_idx}). Effective range: [{start_idx}, {end_idx})."
  )
  print(f"Processing {len(target_dataset)} rows.")
  print(f"Output file: {saved_file}")
  print(f"Checkpoint dir: {checkpoint_dir}")

  client = AsyncOpenAI(base_url=_resolved_base_url, api_key=_resolved_api_key)
  try:
    updated_dataset = await generate_and_update(target_dataset, client, checkpoint_dir)
    save_dataset(updated_dataset, saved_file, convert_to_jsonl=True)

    if os.path.isdir(checkpoint_dir):
      shutil.rmtree(checkpoint_dir)
    print(f"Final dataset saved to {saved_file}.")
  finally:
    await client.close()


if __name__ == "__main__":
  asyncio.run(main())
