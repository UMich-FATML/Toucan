import argparse
import copy
import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Optional, Union

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils import clean_html_comments, clean_json_object, create_preview_json

try:
  from jsonschema import Draft202012Validator, ValidationError as JsonSchemaValidationError
except ImportError:
  print(
    "Error: jsonschema is required for step1.3_process_onet_completion.py. "
    "Install it with: pip install jsonschema"
  )
  raise SystemExit(1)

BAD_PATTERNS = (
  # "i cannot",
  # "i can't",
  # "i'm unable",
  # "i apologize",
  # "i'm sorry",
  "bad_document",
  # "please provide",
  # "could you please",
  # "i need more information",
)
MIN_QUESTION_LEN = 10
SUPPORTED_MODE = "onet_tasks"
SHARDED_RESULTS_FILENAME_RE = re.compile(
  r"^(?P<prefix>.+)_results_(?P<start>\d+)_(?P<end>\d+)\.jsonl$"
)


@dataclass
class ParsedAssistantPayload:
  tool_analysis: str
  cross_tool_workflow: str
  target_tools_str: str
  question: str
  target_tools: list[Any]
  withheld_info: Optional[list[Any]] = None
  target_followup_questions: Optional[list[str]] = None


@dataclass
class SanitizeMetrics:
  min_distance: float
  duplicate_count: int
  min_similar_row_id: Optional[Union[str, int]]


################
# Configurations
################
def get_default_output_schema_path():
  script_dir = os.path.dirname(os.path.abspath(__file__))
  return os.path.join(script_dir, "prompts", "genq_from_onet_tasks_output_schema.json")


def get_args():
  parser = argparse.ArgumentParser(
    description="Tool Use Question Processing and Sanitization Manager (onet_tasks mode only)."
  )

  parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help=(
      "Path to the input JSONL file with model responses. "
      "If missing and ending with '_results.jsonl', the script will try merging "
      "contiguous '*_results_<start>_<end>.jsonl' shards from the same directory."
    ),
  )
  parser.add_argument(
    "--output_schema_file",
    type=str,
    default=get_default_output_schema_path(),
    help="Path to output JSON schema used to validate assistant responses.",
  )

  parser.add_argument(
    "--sentence_model",
    type=str,
    default="sentence-transformers/all-mpnet-base-v2",
    help="SentenceTransformer model for encoding questions.",
  )
  parser.add_argument(
    "--encoding_batch_size",
    type=int,
    default=256,
    help="Batch size for encoding sentences.",
  )
  parser.add_argument(
    "--distance_threshold",
    type=float,
    default=0.1,
    help="Cosine similarity threshold for filtering similar questions.",
  )
  parser.add_argument(
    "--search_space_size",
    type=int,
    default=100,
    help="Number of nearest neighbors to search for similarity.",
  )
  parser.add_argument(
    "--search_batch_size",
    type=int,
    default=4096,
    help="Batch size for searching similarity.",
  )

  parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to run the model on ('cuda' or 'cpu').",
  )
  parser.add_argument(
    "--disable_sanitize",
    action="store_true",
    help="Disable the sanitize step and use extracted questions directly.",
  )
  parser.add_argument(
    "--disable_prepare",
    action="store_true",
    help="Disable the prepare step and stop after sanitization.",
  )
  parser.add_argument(
    "--enable_tool_hint",
    action="store_true",
    help="Enable tool hints by appending tool usage information to the end of each question.",
  )

  return parser.parse_args()


args = get_args()

if args.disable_sanitize and args.disable_prepare:
  print(
    "Warning: Both --disable_sanitize and --disable_prepare are set. "
    "--disable_prepare will be ignored since sanitization is already disabled."
  )

print(
  "Tool Use Question Processing and Sanitization Manager (onet_tasks mode only).\n"
  f"Arguments:\n{args}"
)


################
# Utility Functions
################


def load_output_schema_validator(output_schema_file):
  if not os.path.exists(output_schema_file):
    raise FileNotFoundError(f"Output schema file not found: {output_schema_file}")

  with open(output_schema_file, "r", encoding="utf-8") as f:
    schema = json.load(f)

  if not isinstance(schema, dict):
    raise ValueError("Output schema must be a JSON object.")

  Draft202012Validator.check_schema(schema)
  return Draft202012Validator(schema)


def discover_contiguous_result_shards(input_file):
  if not input_file.endswith("_results.jsonl"):
    raise FileNotFoundError(
      f"Input file not found: {input_file}. "
      "Automatic shard merge is only supported for missing '*_results.jsonl' targets."
    )

  input_dir = os.path.dirname(input_file) or "."
  if not os.path.isdir(input_dir):
    raise FileNotFoundError(
      f"Input directory does not exist: {input_dir}"
    )

  input_basename = os.path.basename(input_file)
  expected_prefix = input_basename[: -len("_results.jsonl")]

  shard_entries = []
  for file_name in os.listdir(input_dir):
    match = SHARDED_RESULTS_FILENAME_RE.fullmatch(file_name)
    if match is None:
      continue
    if match.group("prefix") != expected_prefix:
      continue

    start_idx = int(match.group("start"))
    end_idx = int(match.group("end"))
    shard_entries.append(
      (start_idx, end_idx, os.path.join(input_dir, file_name))
    )

  if not shard_entries:
    raise FileNotFoundError(
      f"Input file not found: {input_file}. "
      f"No matching shards found in {input_dir} for prefix '{expected_prefix}_results_<start>_<end>.jsonl'."
    )

  shard_entries.sort(key=lambda x: x[0])

  for i, (start_idx, end_idx, shard_path) in enumerate(shard_entries):
    if start_idx >= end_idx:
      raise ValueError(
        f"Invalid shard range in {shard_path}: start ({start_idx}) must be smaller than end ({end_idx})."
      )

    if i == 0:
      continue

    prev_start, prev_end, prev_path = shard_entries[i - 1]
    if start_idx < prev_end:
      raise ValueError(
        "Overlapping shard ranges detected: "
        f"{prev_path} ({prev_start}, {prev_end}) and {shard_path} ({start_idx}, {end_idx})."
      )
    if start_idx > prev_end:
      raise ValueError(
        "Gap in shard ranges detected: "
        f"{prev_path} ends at {prev_end}, but next shard {shard_path} starts at {start_idx}."
      )

  return shard_entries


def merge_result_shards(shard_entries, merged_output_file):
  output_dir = os.path.dirname(merged_output_file) or "."
  os.makedirs(output_dir, exist_ok=True)

  line_count = 0
  with open(merged_output_file, "w", encoding="utf-8") as f_out:
    for _, _, shard_path in shard_entries:
      with open(shard_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
          if not line.strip():
            continue
          f_out.write(line if line.endswith("\n") else f"{line}\n")
          line_count += 1

  return line_count


def resolve_or_merge_input_file(input_file):
  if os.path.exists(input_file):
    print(f"Using existing input file: {input_file}")
    return input_file

  print(f"Input file not found: {input_file}. Trying contiguous shard fallback...")
  shard_entries = discover_contiguous_result_shards(input_file)
  start_idx = shard_entries[0][0]
  end_idx = shard_entries[-1][1]
  merged_lines = merge_result_shards(shard_entries, input_file)

  print(
    f"Merged {len(shard_entries)} shard file(s) into {input_file}. "
    f"Contiguous range: [{start_idx}, {end_idx}). Non-empty rows written: {merged_lines}."
  )
  return input_file


def filter_metadata_by_target_tools(metadata, target_tools_str):
  """
  Filter metadata to only include MCP servers that provide the tools mentioned in target_tools.
  This reduces file size by removing server info that's not actually used for the question.
  Only applies filtering when multi_server_allocation_strategy is random_featured.
  """
  copied_metadata = copy.deepcopy(metadata) if isinstance(metadata, dict) else metadata
  if not isinstance(copied_metadata, dict):
    return metadata

  if not (isinstance(target_tools_str, str) and target_tools_str.strip()):
    return copied_metadata

  target_tools_raw = [t.strip() for t in target_tools_str.split(",") if t.strip()]
  if not target_tools_raw:
    return copied_metadata

  if (
    copied_metadata.get("question_gen_args", {}).get("multi_server_allocation_strategy", "")
    != "random_featured"
  ):
    return copied_metadata

  servers = copied_metadata.get("mcp_servers")
  if not isinstance(servers, list):
    return copied_metadata

  server_tool_combos = set()
  for tool_entry in target_tools_raw:
    if "::" not in tool_entry:
      raise ValueError(
        "All target tools must be specified in 'server_name::tool_name' format. "
        f"Found tool entry without server: '{tool_entry}'."
      )
    server_name, tool_name = tool_entry.split("::", 1)
    server_tool_combos.add((server_name.strip(), tool_name.strip()))

  filtered_servers = []
  for server_info in servers:
    server_name = server_info.get("server_name", "Unknown Server")
    remote_response = server_info.get("remote_server_response", {})
    server_tools = remote_response.get("tools", [])

    has_matching_tools = any(
      (server_name, tool.get("name", "")) in server_tool_combos for tool in server_tools
    )
    if has_matching_tools:
      filtered_servers.append(server_info)

  copied_metadata["mcp_servers"] = filtered_servers
  if "server_count" in copied_metadata:
    copied_metadata["server_count"] = len(filtered_servers)

  return copied_metadata


def prune_metadata_for_output(metadata):
  """
  Remove bulky per-server metadata fields to reduce artifact size.
  """
  copied_metadata = copy.deepcopy(metadata) if isinstance(metadata, dict) else metadata
  if not isinstance(copied_metadata, dict):
    return metadata

  mcp_servers = copied_metadata.get("mcp_servers")
  if not isinstance(mcp_servers, list):
    return copied_metadata

  for server_data in mcp_servers:
    server_info = server_data.get("server_info")
    if isinstance(server_info, dict):
      server_info.pop("file_path", None)
      server_info.pop("tools", None)
      server_info.pop("tools_count", None)

  return copied_metadata


def extract_json_payload_candidates(response_content):
  """
  Build ordered JSON parsing candidates:
  1) Last fenced ```json ... ``` block (if present).
  2) Full assistant content.
  """
  if not isinstance(response_content, str):
    return []

  text = response_content.strip()
  if not text:
    return []

  candidates = []
  fenced_matches = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
  if fenced_matches:
    candidates.append(("fenced JSON response", fenced_matches[-1].strip()))

  candidates.append(("JSON response", text))

  deduped_candidates = []
  seen_payloads = set()
  for source_label, candidate in candidates:
    if not candidate or candidate in seen_payloads:
      continue
    deduped_candidates.append((source_label, candidate))
    seen_payloads.add(candidate)

  return deduped_candidates


def parse_json_response(response_content, output_schema_validator, metadata=None):
  """
  Parse the JSON response from the assistant to extract tool_analysis,
  cross_tool_workflow, target_tools, and request.

  Args:
    response_content: Assistant response text.
    output_schema_validator: JSON schema validator for assistant payloads.
    metadata: Optional row metadata.
  """
  candidate_payloads = extract_json_payload_candidates(response_content)
  if not candidate_payloads:
    return None

  parse_errors = []
  schema_errors = []

  for source_label, candidate in candidate_payloads:
    try:
      parsed_json = json.loads(candidate)
    except json.JSONDecodeError as e:
      parse_errors.append((source_label, str(e)))
      continue

    if not isinstance(parsed_json, dict):
      parse_errors.append((source_label, "Parsed JSON payload is not an object."))
      continue

    try:
      output_schema_validator.validate(parsed_json)
    except JsonSchemaValidationError as e:
      schema_errors.append((source_label, e.message))
      continue

    return extract_individual_components(parsed_json, metadata)

  for source_label, message in parse_errors:
    if source_label == "fenced JSON response":
      print(f"Error parsing fenced JSON response: {message}")
    else:
      print(f"Error parsing JSON response: {message}")

  for source_label, message in schema_errors:
    print(f"Output schema validation failed for {source_label}: {message}")

  return None


def extract_individual_components(parsed_json, metadata=None):
  """
  Extract components from JSON for onet_tasks mode.
  """
  tool_analysis = parsed_json.get("tool_analysis", "")
  cross_tool_workflow = parsed_json.get("cross_tool_workflow", "")
  request_text = parsed_json.get("request", "")

  target_tools_array = parsed_json.get("target_tools", None)
  if not isinstance(target_tools_array, list):
    return None

  target_tools_str = extract_json_tools(target_tools_array, metadata)

  if not all([tool_analysis, target_tools_str, request_text]):
    return None

  withheld_info = parsed_json.get("withheld_info", None)
  if not isinstance(withheld_info, list) or len(withheld_info) == 0:
    withheld_info = None

  target_followup_questions = parsed_json.get("target_followup_questions", None)
  if not isinstance(target_followup_questions, list) or len(target_followup_questions) == 0:
    target_followup_questions = None

  payload = ParsedAssistantPayload(
    tool_analysis=tool_analysis.strip(),
    cross_tool_workflow=cross_tool_workflow.strip() if cross_tool_workflow else "",
    target_tools_str=target_tools_str.strip(),
    question=clean_html_comments(request_text.strip()),
    target_tools=target_tools_array,
    withheld_info=withheld_info,
    target_followup_questions=target_followup_questions,
  )
  return payload


def extract_json_tools(target_tools_array, metadata=None):
  """
  Extract target tools from JSON array format for onet_tasks mode.
  Expected format: [{"server": "Server1", "tool": "search", "arguments": {...}}, ...]
  """
  if not isinstance(target_tools_array, list) or len(target_tools_array) == 0:
    return ""

  tools_list = []
  for tool_obj in target_tools_array:
    if not isinstance(tool_obj, dict):
      continue

    server_name = str(tool_obj.get("server", "")).strip()
    tool_name = str(tool_obj.get("tool", "")).strip()
    if not server_name or not tool_name:
      continue

    tools_list.append(f"{server_name}::{tool_name}")

  return ", ".join(tools_list)


def find_last_nonempty_assistant_message(messages):
  for msg in reversed(messages):
    if msg.get("role") == "assistant":
      content = msg.get("content", "")
      if isinstance(content, str) and content.strip():
        return content
  return None


def contains_bad_pattern(question):
  q = question.lower()
  return any(pattern in q for pattern in BAD_PATTERNS)


def is_valid_question_text(question):
  if not isinstance(question, str):
    return False
  stripped = question.strip()
  return bool(stripped) and len(stripped) >= MIN_QUESTION_LEN and not contains_bad_pattern(stripped)


def extract_questions(input_file, output_file, output_schema_validator, preview_file=None):
  """
  Extract structured questions from assistant responses with JSON format for onet_tasks mode.

  Args:
    input_file: Input JSONL path with model responses.
    output_file: Output JSONL path for extracted records.
    output_schema_validator: JSON schema validator for assistant payloads.
    preview_file: Optional preview JSON output path.
  """
  total_processed = 0
  successfully_parsed = 0
  mode_counts = {SUPPORTED_MODE: 0}

  with open(input_file, "r", encoding="utf-8") as f_in, open(
    output_file, "w", encoding="utf-8"
  ) as f_out:
    for line in tqdm(f_in, desc="Extracting Questions"):
      try:
        data = json.loads(line)
        messages = data.get("messages", [])
        metadata = data.get("metadata", {})
        mode_counts[SUPPORTED_MODE] += 1

        assistant_content = find_last_nonempty_assistant_message(messages)
        if not assistant_content:
          print("No non-empty assistant message found. Skipping.")
          continue

        total_processed += 1
        parsed_response = parse_json_response(
          assistant_content, output_schema_validator, metadata
        )
        if not parsed_response:
          print(f"Failed to parse JSON response for row {total_processed}. Skipping.")
          continue

        if not parsed_response.target_tools_str.strip():
          print(f"No target tools extracted for row {total_processed}. Skipping.")
          continue

        if not is_valid_question_text(parsed_response.question):
          print(f"Question validation failed for row {total_processed}. Skipping.")
          continue

        filtered_metadata = filter_metadata_by_target_tools(
          metadata, parsed_response.target_tools_str
        )
        filtered_metadata = prune_metadata_for_output(filtered_metadata)

        result = {
          "target_tools": parsed_response.target_tools,
          "target_tools_str": parsed_response.target_tools_str,
          "question": parsed_response.question,
          "tool_analysis": parsed_response.tool_analysis,
          "cross_tool_workflow": parsed_response.cross_tool_workflow,
          "metadata": {
            **filtered_metadata,
            "server_count": get_server_count(filtered_metadata),
          },
        }

        if parsed_response.withheld_info is not None:
          result["withheld_info"] = parsed_response.withheld_info
        if parsed_response.target_followup_questions is not None:
          result["target_followup_questions"] = parsed_response.target_followup_questions

        result = clean_json_object(result)
        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
        successfully_parsed += 1

      except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        continue
      except Exception as e:
        print(f"Unexpected error processing line: {e}")
        continue

  print(
    "Finished extracting questions. "
    f"Total processed: {total_processed}, Successfully parsed: {successfully_parsed}"
  )
  print(f"Mode distribution: {mode_counts}")
  print(f"Output saved to {output_file}")

  if preview_file:
    create_preview_json(output_file, preview_file)


def get_server_count(metadata):
  """
  Get the number of servers involved based on the metadata.
  """
  mcp_servers = metadata.get("mcp_servers", [])
  return len(mcp_servers) if isinstance(mcp_servers, list) else 0


def build_sanitized_entry(item, metric):
  """
  Build a sanitized/distance entry with consistent schema.
  """
  target_tools = item.get("target_tools")
  if not isinstance(target_tools, list):
    raise ValueError("target_tools is required and must be a list.")

  target_tools_str = item.get("target_tools_str")
  if not isinstance(target_tools_str, str) or not target_tools_str.strip():
    target_tools_str = extract_json_tools(target_tools)
  if not target_tools_str:
    raise ValueError("target_tools_str is required and must be derivable from target_tools.")

  filtered_metadata = filter_metadata_by_target_tools(item.get("metadata", {}), target_tools_str)
  filtered_metadata = prune_metadata_for_output(filtered_metadata)

  entry = {
    "target_tools": target_tools,
    "target_tools_str": target_tools_str,
    "question": item["question"],
    "metadata": filtered_metadata,
    "min_distance": metric.min_distance,
    "duplicate_count": metric.duplicate_count,
    "min_similar_row_id": metric.min_similar_row_id,
  }

  if item.get("tool_analysis"):
    entry["tool_analysis"] = item["tool_analysis"]

  workflow = item.get("cross_tool_workflow")
  if workflow:
    entry["cross_tool_workflow"] = workflow

  if item.get("withheld_info"):
    entry["withheld_info"] = item["withheld_info"]

  if item.get("target_followup_questions"):
    entry["target_followup_questions"] = item["target_followup_questions"]

  return clean_json_object(entry)


def sanitize_questions(
  input_file,
  sanitized_output,
  distance_output,
  preview_distance_file,
  preview_sanitized_file,
  sentence_model,
  encoding_batch_size,
  distance_threshold,
  search_space_size,
  search_batch_size,
  device,
):
  """
  Sanitize questions by removing duplicates based on semantic similarity.
  """
  print(f"Loading dataset from {input_file}...")
  dataset_items = []
  with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      dataset_items.append(json.loads(line))

  questions = [item["question"] for item in dataset_items]
  print(f"Number of questions: {len(questions)}")

  if len(questions) == 0:
    print("No questions found. Writing empty outputs.")
    open(distance_output, "w", encoding="utf-8").close()
    open(sanitized_output, "w", encoding="utf-8").close()
    if preview_distance_file:
      create_preview_json(distance_output, preview_distance_file)
    if preview_sanitized_file:
      create_preview_json(sanitized_output, preview_sanitized_file)
    return

  print("Loading SentenceTransformer model...")
  model = SentenceTransformer(sentence_model)
  model.to(device)

  print("Encoding questions into embeddings...")
  embeddings = model.encode(
    questions,
    batch_size=encoding_batch_size,
    convert_to_numpy=True,
    show_progress_bar=True,
  )
  print(f"Embeddings shape: {embeddings.shape}")

  print("Building Faiss index (CPU only)...")
  dimension = embeddings.shape[1]
  faiss_index = faiss.IndexFlatL2(dimension)
  faiss_index.add(embeddings)
  print(f"Faiss index has {faiss_index.ntotal} vectors.")

  print("Searching for similar questions...")
  batch_size = search_batch_size
  k = min(search_space_size + 1, len(embeddings))
  similar_indices = []
  similar_scores = []
  for i in tqdm(range(0, len(embeddings), batch_size), desc="Searching Batches"):
    end = min(i + batch_size, len(embeddings))
    scores, indices = faiss_index.search(embeddings[i:end], k)
    similar_indices.append(indices)
    similar_scores.append(scores)

  similar_indices = np.vstack(similar_indices)
  similar_scores = np.vstack(similar_scores)

  print("Applying similarity threshold...")
  row_ids = []
  for item in dataset_items:
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
      row_ids.append(metadata.get("row_id", None))
    else:
      row_ids.append(None)

  metrics = []
  for idx in tqdm(range(len(questions)), desc="Processing Questions"):
    similar = similar_indices[idx]
    scores = similar_scores[idx]

    self_matches = np.where(similar == idx)[0]
    if len(self_matches) > 0:
      self_idx = int(self_matches[0])
      similar_filtered = np.delete(similar, self_idx)
      scores_filtered = np.delete(scores, self_idx)
    else:
      similar_filtered = similar
      scores_filtered = scores

    duplicate_count = int(np.sum(scores_filtered < distance_threshold))
    min_distance = (
      float(scores_filtered[np.argmin(scores_filtered)])
      if len(scores_filtered) > 0
      else float("inf")
    )
    min_similar_row_id = (
      row_ids[int(similar_filtered[np.argmin(scores_filtered)])]
      if len(scores_filtered) > 0
      else row_ids[idx]
    )

    metrics.append(
      SanitizeMetrics(
        min_distance=min_distance,
        duplicate_count=duplicate_count,
        min_similar_row_id=min_similar_row_id,
      )
    )

  print("Saving sanitized questions...")
  total_rows = 0
  with open(distance_output, "w", encoding="utf-8") as f_out:
    for idx, item in enumerate(tqdm(dataset_items, desc="Preparing all entries")):
      entry = build_sanitized_entry(item, metrics[idx])
      f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
      total_rows += 1
  print(f"Wrote {total_rows} rows to {distance_output}")

  filtered_rows = 0
  with open(sanitized_output, "w", encoding="utf-8") as f_out:
    for idx, item in enumerate(tqdm(dataset_items, desc="Preparing filtered entries")):
      metric = metrics[idx]
      if metric.min_distance > distance_threshold or metric.min_similar_row_id == row_ids[idx]:
        entry = build_sanitized_entry(item, metric)
        f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        filtered_rows += 1
  print(f"Wrote {filtered_rows} rows to {sanitized_output}")
  print(f"Sanitized questions saved to {sanitized_output}")
  print("Sanitization process completed.")

  if preview_distance_file:
    create_preview_json(distance_output, preview_distance_file)
  if preview_sanitized_file:
    create_preview_json(sanitized_output, preview_sanitized_file)


def prepare_questions(input_file, output_file):
  """
  Prepare questions for final use by creating the proper message format.
  Handles onet_tasks mode only.
  """
  print(f"Preparing questions from {input_file}")
  stats = {
    "total_questions": 0,
    SUPPORTED_MODE: 0,
    "server_count_distribution": {},
    "allocation_strategies": {},
  }

  with open(input_file, "r", encoding="utf-8") as f, open(
    output_file, "w", encoding="utf-8"
  ) as outf:
    for line in tqdm(f, desc="Preparing Questions"):
      data = json.loads(line)
      metadata = data.get("metadata", {})

      target_tools = data.get("target_tools")
      if not isinstance(target_tools, list):
        raise ValueError("target_tools is required and must be a list.")

      target_tools_str = data.get("target_tools_str")
      if not isinstance(target_tools_str, str) or not target_tools_str.strip():
        target_tools_str = extract_json_tools(target_tools)
      if not target_tools_str:
        raise ValueError("target_tools_str is required and must be derivable from target_tools.")

      filtered_metadata = filter_metadata_by_target_tools(metadata, target_tools_str)
      filtered_metadata = prune_metadata_for_output(filtered_metadata)

      stats["total_questions"] += 1
      stats[SUPPORTED_MODE] += 1

      server_count = filtered_metadata.get("server_count", get_server_count(filtered_metadata))
      stats["server_count_distribution"][str(server_count)] = (
        stats["server_count_distribution"].get(str(server_count), 0) + 1
      )

      question_content = data["question"]
      if args.enable_tool_hint and target_tools_str:
        question_content = (
          f"{data['question']}\n\n"
          f"You need to solve this question using {target_tools_str} tool "
          "from the list of available tools."
        )

      result = {
        "messages": [{"role": "user", "content": question_content}],
        "metadata": {
          **filtered_metadata,
          "target_tools": target_tools,
          "target_tools_str": target_tools_str,
          "question": data["question"],
          "min_distance": data.get("min_distance", None),
          "duplicate_count": data.get("duplicate_count", 0),
          "min_similar_row_id": data.get("min_similar_row_id", None),
        },
      }

      if data.get("tool_analysis"):
        result["metadata"]["tool_analysis"] = data["tool_analysis"]
      if data.get("cross_tool_workflow"):
        result["metadata"]["cross_tool_workflow"] = data["cross_tool_workflow"]
      if data.get("withheld_info"):
        result["metadata"]["withheld_info"] = data["withheld_info"]
      if data.get("target_followup_questions"):
        result["metadata"]["target_followup_questions"] = data["target_followup_questions"]

      result = clean_json_object(result)
      outf.write(json.dumps(result, ensure_ascii=False) + "\n")

  stats_file = output_file.replace(".jsonl", "_stats.json")
  with open(stats_file, "w", encoding="utf-8") as stats_outf:
    json.dump(stats, stats_outf, ensure_ascii=False, indent=2)

  print(f"Finished preparing questions. Output saved to {output_file}")
  print(f"Statistics saved to {stats_file}")
  print_processing_summary(stats)


def print_processing_summary(stats):
  """
  Print a summary of the processing results for onet_tasks mode.
  """
  print("\n" + "=" * 60)
  print("PROCESSING SUMMARY")
  print("=" * 60)

  print(f"Total Questions Processed: {stats['total_questions']}")
  print(f"{SUPPORTED_MODE} Questions: {stats[SUPPORTED_MODE]}")

  if stats[SUPPORTED_MODE] > 0:
    print(f"\n{SUPPORTED_MODE} Statistics:")
    print("-" * 30)

    if stats["server_count_distribution"]:
      print("Server Count Distribution:")
      for count, freq in sorted(stats["server_count_distribution"].items()):
        percentage = (freq / stats[SUPPORTED_MODE]) * 100
        print(f"  {count} servers: {freq} questions ({percentage:.1f}%)")

  print("=" * 60 + "\n")


def main():
  output_schema_validator = load_output_schema_validator(args.output_schema_file)
  print(f"Loaded output schema from: {args.output_schema_file}")

  print(f"Tool Use Question Processing Pipeline ({SUPPORTED_MODE} mode only). Arguments: {args}")
  resolved_input_file = resolve_or_merge_input_file(args.input_file)

  input_dir = os.path.dirname(resolved_input_file) or "."
  input_basename = os.path.basename(resolved_input_file)
  print(f"Input directory: {input_dir}")
  print(f"Input basename: {input_basename}")

  output_path = f"{input_dir}/processed"
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  base_name = input_basename.replace("_results.jsonl", "")
  extracted_output = f"{output_path}/{base_name}_1extracted.jsonl"
  extracted_output_review = f"{output_path}/preview_{base_name}_1extracted.json"
  distance_output = f"{output_path}/{base_name}_2distance.jsonl"
  distance_output_review = f"{output_path}/preview_{base_name}_2distance.json"
  sanitized_output = f"{output_path}/{base_name}_3sanitized.jsonl"
  sanitized_output_review = f"{output_path}/preview_{base_name}_3sanitized.json"
  prepared_output = f"{output_path}/{base_name}_4prepared.jsonl"
  prepared_output_review = f"{output_path}/preview_{base_name}_4prepared.json"

  print("Step 1: Extracting questions from JSON responses...")
  extract_questions(
    resolved_input_file, extracted_output, output_schema_validator, extracted_output_review
  )

  if not args.disable_sanitize:
    print("Step 2: Sanitizing questions (removing duplicates)...")
    sanitize_questions(
      input_file=extracted_output,
      sanitized_output=sanitized_output,
      distance_output=distance_output,
      preview_distance_file=distance_output_review,
      preview_sanitized_file=sanitized_output_review,
      sentence_model=args.sentence_model,
      encoding_batch_size=args.encoding_batch_size,
      distance_threshold=args.distance_threshold,
      search_space_size=args.search_space_size,
      search_batch_size=args.search_batch_size,
      device=args.device,
    )
    input_for_prepare = sanitized_output
  else:
    print("Sanitize step is disabled. Using extracted questions directly.")
    shutil.copyfile(extracted_output, sanitized_output)
    create_preview_json(sanitized_output, sanitized_output_review)
    input_for_prepare = sanitized_output

  if not args.disable_prepare:
    print("Step 3: Preparing questions for final use...")
    if args.enable_tool_hint:
      print(
        "Tool hints enabled: Appending "
        "'You need to solve this question using {target_tools}.' to questions."
      )
    prepare_questions(input_for_prepare, prepared_output)
    create_preview_json(prepared_output, prepared_output_review)
    print(f"Final output saved to: {prepared_output}")
  else:
    print("Prepare step is disabled. Stopping after sanitization.")
    print(f"Final output saved to: {input_for_prepare}")


if __name__ == "__main__":
  main()
