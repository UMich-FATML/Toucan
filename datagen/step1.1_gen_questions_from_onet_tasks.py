import argparse
import itertools
import json
import os
import random
import re
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import load_jsonl_to_list

"""
This script generates task-first tool-use prompts from O*NET tasks.

Deterministic generation flow:
1. Load tasks with matched MCP servers from tasks_to_smithery_servers.jsonl
2. Load occupation metadata and MCP server metadata
3. Filter to tasks that have loadable matched servers
4. Enumerate deterministic (occupation, task_combination, server_assignment) tuples
5. Generate prompts for all tuples, or the first N tuples if --total_prompts is set
6. Save:
   - generated prompts JSONL
   - combinations parquet (the considered combinations)
   - generation args JSON

Example:
  python step1.1_gen_questions_from_onet_tasks.py --num_tasks 2 --total_prompts 100
"""


@dataclass
class InputBundle:
  tasks_list: list[dict[str, Any]]
  occupation_dict: dict[str, dict[str, str]]
  prompt_template: str
  server_index: dict[str, dict[str, Any]]


@dataclass
class GenerationPool:
  tasks_list_filtered: list[dict[str, Any]]
  occupation_to_tasks: dict[str, list[dict[str, Any]]]
  valid_onet_codes: list[str]


@dataclass
class OutputPaths:
  output_file_path: str
  output_base_dir: str
  combos_parquet_path: str
  args_file_path: str


@dataclass
class ServerContext:
  tools_by_server: dict[str, list[dict[str, Any]]]
  matched_servers_meta: list[dict[str, str]]
  mcp_servers_metadata: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Tool Use Question Generation using O*NET task-first approach.")

  parser.add_argument("--num_tasks", type=int, required=True, help="Number of O*NET tasks per prompt.")
  parser.add_argument(
    "--num_tools",
    type=int,
    default=None,
    help="Minimum number of tools required in model output; defaults to --num_tasks.",
  )
  parser.add_argument(
    "--total_prompts",
    type=int,
    default=None,
    help="Total prompts to generate. If omitted, generate all available combinations.",
  )

  parser.add_argument("--output_folder", type=str, default="../data", help="Output folder path.")
  parser.add_argument("--job_name", type=str, default=None, help="Job name for organization.")
  parser.add_argument("--timestamp", type=int, default=int(time.time()), help="Timestamp for output naming.")
  parser.add_argument("--seed", type=int, default=None, help="Random seed.")
  parser.add_argument(
    "--mcp_servers_dir",
    type=str,
    default="../mcp_servers/smithery_mcp_servers_0210",
    help="Directory containing MCP server JSON files.",
  )
  parser.add_argument(
    "--no_refs",
    action="store_true",
    help="Disable loading task references from task_refs.parquet and use N/A references.",
  )

  args = parser.parse_args()
  if args.num_tools is None:
    args.num_tools = args.num_tasks
  return args


def resolve_paths(script_dir: str, no_refs: bool) -> dict[str, str]:
  prompt_template_filename = "genq_from_onet_tasks_no_refs.md" if no_refs else "genq_from_onet_tasks.md"
  return {
    "tasks_to_servers_path": os.path.join(script_dir, "tasks_to_smithery_servers.jsonl"),
    "occupation_data_path": os.path.join(script_dir, "onet_db_30_1_text", "Occupation Data.txt"),
    "prompt_template_path": os.path.join(script_dir, "prompts", prompt_template_filename),
    "task_refs_path": os.path.join(script_dir, "task_refs.parquet"),
  }


def load_occupation_data(occupation_file_path: str) -> dict[str, dict[str, str]]:
  occupations: dict[str, dict[str, str]] = {}
  with open(occupation_file_path, "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
      parts = line.strip().split("\t")
      if len(parts) >= 3:
        code = parts[0]
        title = parts[1]
        description = parts[2]
        occupations[code] = {"title": title, "description": description}
  print(f"Loaded {len(occupations)} occupations from {occupation_file_path}")
  return occupations


def load_prompt_template(template_path: str) -> str:
  with open(template_path, "r", encoding="utf-8") as f:
    return f.read()


def normalize_task_ref_results(results: Any) -> list[Any]:
  if results is None:
    return []
  if isinstance(results, list):
    return results
  if isinstance(results, tuple):
    return list(results)
  if isinstance(results, np.ndarray):
    return results.tolist()

  try:
    if pd.isna(results):
      return []
  except Exception:
    pass

  return []


def load_task_refs_index(task_refs_path: str) -> dict[tuple[str, str], dict[str, Any]]:
  if not os.path.exists(task_refs_path):
    raise FileNotFoundError(f"Task refs file not found: {task_refs_path}")

  task_refs_df = pd.read_parquet(task_refs_path)
  required_columns = {"onet_soc_code", "task_id", "query", "results"}
  missing_columns = sorted(required_columns - set(task_refs_df.columns))
  if missing_columns:
    raise ValueError(f"Task refs parquet missing required columns: {missing_columns}")

  task_refs_index: dict[tuple[str, str], dict[str, Any]] = {}
  duplicate_keys = 0
  skipped_rows = 0

  for row_idx, row in task_refs_df.iterrows():
    onet_code = str(row.get("onet_soc_code", "")).strip()
    task_id = str(row.get("task_id", "")).strip()
    if not onet_code or not task_id:
      skipped_rows += 1
      continue

    key = (onet_code, task_id)
    if key in task_refs_index:
      duplicate_keys += 1
      continue

    query_value = row.get("query", "")
    query = "" if query_value is None else str(query_value).strip()
    results = normalize_task_ref_results(row.get("results", []))

    task_refs_index[key] = {
      "query": query,
      "results": results,
      "row_idx": int(row_idx),
    }

  print(
    f"Loaded {len(task_refs_index)} task refs from {task_refs_path} "
    f"({duplicate_keys} duplicate keys skipped, {skipped_rows} invalid rows skipped)"
  )
  return task_refs_index


@dataclass
class TaskRefLookupStats:
  requested: int = 0
  hits: int = 0
  misses: int = 0
  empty_results: int = 0
  _lock: Lock = field(default_factory=Lock, repr=False)

  def record(self, found: bool, has_non_empty_result: bool) -> None:
    with self._lock:
      self.requested += 1
      if not found:
        self.misses += 1
      elif has_non_empty_result:
        self.hits += 1
      else:
        self.empty_results += 1


def create_occupation_to_tasks_index(tasks_list: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
  occupation_to_tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
  for task in tasks_list:
    matched_servers = task.get("matched_servers", [])
    if matched_servers:
      onet_code = task.get("onet_soc_code")
      if onet_code:
        occupation_to_tasks[onet_code].append(task)
  print(f"Built index with {len(occupation_to_tasks)} occupations (tasks with matched servers)")
  return occupation_to_tasks


def create_server_metadata_index(tasks_list: list[dict[str, Any]], mcp_servers_dir: str) -> dict[str, dict[str, Any]]:
  server_ids = set()
  for task in tasks_list:
    for server in task.get("matched_servers", []):
      sid = server.get("server_id")
      if sid:
        server_ids.add(sid)

  print(f"Found {len(server_ids)} unique server IDs to load")

  server_index: dict[str, dict[str, Any]] = {}
  missing = 0
  skipped_validation = 0

  for server_id in sorted(server_ids):
    file_path = os.path.join(mcp_servers_dir, f"{server_id}.json")
    if not os.path.exists(file_path):
      missing += 1
      continue
    try:
      with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
      validation_error = data.get("server", {}).get("validation_error")
      if validation_error is not None:
        skipped_validation += 1
        continue
      server_index[server_id] = data
    except (json.JSONDecodeError, OSError) as exc:
      print(f"Warning: Could not load {file_path}: {exc}")
      missing += 1

  print(
    f"Loaded {len(server_index)} server metadata files "
    f"({missing} missing/failed, {skipped_validation} skipped due to validation errors)"
  )
  return server_index


def filter_tasks_with_loadable_servers(
  tasks_list: list[dict[str, Any]],
  server_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
  valid_server_ids = set(server_index.keys())
  filtered = []
  dropped_no_servers = 0
  dropped_no_code = 0

  for task in tasks_list:
    onet_code = task.get("onet_soc_code")
    if not onet_code:
      dropped_no_code += 1
      continue

    valid_matched_servers = []
    for server in task.get("matched_servers", []):
      server_id = server.get("server_id")
      if server_id in valid_server_ids:
        valid_matched_servers.append(server)

    if not valid_matched_servers:
      dropped_no_servers += 1
      continue

    task_copy = dict(task)
    task_copy["matched_servers"] = valid_matched_servers
    filtered.append(task_copy)

  print(
    "Filtered tasks to loadable matched servers: "
    f"{len(filtered)} kept, {dropped_no_servers} dropped (no loadable servers), "
    f"{dropped_no_code} dropped (missing onet code)"
  )
  return filtered


def get_valid_onet_codes(occupation_to_tasks: dict[str, list[dict[str, Any]]], num_tasks: int) -> list[str]:
  valid_onet_codes = [code for code, tasks in occupation_to_tasks.items() if len(tasks) >= num_tasks]
  print(f"Found {len(valid_onet_codes)} occupations with >= {num_tasks} tasks with matched servers")
  return valid_onet_codes


def iter_combo_records(
  occupation_to_tasks: dict[str, list[dict[str, Any]]],
  valid_onet_codes: list[str],
  num_tasks: int,
) -> Iterator[dict[str, Any]]:
  for onet_code in sorted(valid_onet_codes):
    tasks = occupation_to_tasks[onet_code]
    tasks_sorted = sorted(tasks, key=lambda t: t.get("task_id", ""))

    for task_indices in itertools.combinations(range(len(tasks_sorted)), num_tasks):
      selected_tasks = [tasks_sorted[i] for i in task_indices]
      per_task_server_ids = []
      has_empty_server_set = False

      for task in selected_tasks:
        server_ids = sorted(
          {
            server.get("server_id")
            for server in task.get("matched_servers", [])
            if server.get("server_id")
          }
        )
        if not server_ids:
          has_empty_server_set = True
          break
        per_task_server_ids.append(server_ids)

      if has_empty_server_set:
        continue

      for server_assignment in itertools.product(*per_task_server_ids):
        yield {
          "onet_code": onet_code,
          "task_indices": task_indices,
          "tasks": selected_tasks,
          "selected_server_ids": server_assignment,
        }


def count_combo_records(
  occupation_to_tasks: dict[str, list[dict[str, Any]]],
  valid_onet_codes: list[str],
  num_tasks: int,
  limit: int | None = None,
) -> int:
  count = 0
  combo_iter = iter_combo_records(occupation_to_tasks, valid_onet_codes, num_tasks)
  if limit is not None:
    combo_iter = itertools.islice(combo_iter, limit)

  for _ in combo_iter:
    count += 1

  print(f"Built {count} (occupation, task_combo, server_assignment) combinations")
  return count


def compute_total_prompts(requested_total: int | None, combo_count: int) -> int:
  if requested_total is None:
    return combo_count
  if requested_total > combo_count:
    print(f"Warning: Requested {requested_total} prompts but only {combo_count} combinations available.")
    print(f"Generating {combo_count} prompts instead.")
    return combo_count
  return requested_total


def prepare_output_paths(args: argparse.Namespace, total_prompts_tag: str) -> OutputPaths:
  output_dirname = f"onet_tasks_{args.num_tasks}_tasks_{total_prompts_tag}_{args.timestamp}"
  output_base_dir = os.path.join(args.output_folder, args.job_name or output_dirname)
  os.makedirs(output_base_dir, exist_ok=True)

  output_file_path = os.path.join(output_base_dir, f"{output_dirname}_prepared.jsonl")
  combos_parquet_path = os.path.join(output_base_dir, "combos.parquet")
  args_file_path = os.path.join(output_base_dir, "generation_args.json")
  return OutputPaths(output_file_path, output_base_dir, combos_parquet_path, args_file_path)


def save_generation_args(args: argparse.Namespace, args_file_path: str) -> dict[str, Any]:
  args_dict = vars(args)
  with open(args_file_path, "w", encoding="utf-8") as f:
    json.dump(args_dict, f, indent=2)
  print(f"Arguments saved to: {args_file_path}")
  return args_dict


def build_combo_parquet_row(combo_idx: int, combo_record: dict[str, Any]) -> dict[str, Any]:
  return {
    "combo_idx": combo_idx,
    "onet_code": combo_record["onet_code"],
    "task_indices": list(combo_record["task_indices"]),
    "selected_server_ids": list(combo_record["selected_server_ids"]),
    "tasks_metadata": [
      {
        "task_id": task.get("task_id"),
        "task": task.get("task"),
        "matched_servers": [server.get("server_id") for server in task.get("matched_servers", [])],
      }
      for task in combo_record["tasks"]
    ],
  }


def write_empty_combos_parquet(combos_parquet_path: str) -> None:
  try:
    import pyarrow as pa
    import pyarrow.parquet as pq
  except ImportError as exc:
    raise RuntimeError("pyarrow is required to write combos.parquet") from exc

  task_metadata_type = pa.list_(
    pa.struct(
      [
        pa.field("task_id", pa.string()),
        pa.field("task", pa.string()),
        pa.field("matched_servers", pa.list_(pa.string())),
      ]
    )
  )

  empty_table = pa.Table.from_arrays(
    [
      pa.array([], type=pa.int64()),
      pa.array([], type=pa.string()),
      pa.array([], type=pa.list_(pa.int64())),
      pa.array([], type=pa.list_(pa.string())),
      pa.array([], type=task_metadata_type),
    ],
    names=["combo_idx", "onet_code", "task_indices", "selected_server_ids", "tasks_metadata"],
  )
  pq.write_table(empty_table, combos_parquet_path)


def serialize_combos_streaming(
  occupation_to_tasks: dict[str, list[dict[str, Any]]],
  valid_onet_codes: list[str],
  num_tasks: int,
  total_prompts: int,
  combos_parquet_path: str,
  chunk_size: int = 10000,
) -> None:
  try:
    import pyarrow as pa
    import pyarrow.parquet as pq
  except ImportError as exc:
    raise RuntimeError("pyarrow is required to write combos.parquet") from exc

  writer = None
  rows_buffer: list[dict[str, Any]] = []
  written = 0

  def flush_buffer() -> None:
    nonlocal writer
    if not rows_buffer:
      return
    table = pa.Table.from_pylist(rows_buffer)
    if writer is None:
      writer = pq.ParquetWriter(combos_parquet_path, table.schema)
    writer.write_table(table)
    rows_buffer.clear()

  try:
    combo_iter = itertools.islice(
      iter_combo_records(occupation_to_tasks, valid_onet_codes, num_tasks),
      total_prompts,
    )

    for combo_idx, combo_record in enumerate(combo_iter):
      rows_buffer.append(build_combo_parquet_row(combo_idx, combo_record))
      written = combo_idx + 1
      if len(rows_buffer) >= chunk_size:
        flush_buffer()

    flush_buffer()
  finally:
    if writer is not None:
      writer.close()

  if written == 0:
    write_empty_combos_parquet(combos_parquet_path)

  print(f"Combinations dataframe saved to: {combos_parquet_path}")


def init_generation_settings(args: argparse.Namespace) -> int:
  worker_count = 8
  if args.no_refs:
    print("Task refs disabled (--no_refs set). Task references will use N/A.")
  print(f"Using {worker_count} parallel worker threads for prompt generation")
  return worker_count


def format_tasks_list(tasks: list[dict[str, Any]]) -> str:
  lines = []
  for i, task in enumerate(tasks, 1):
    lines.append(f"{i}. {task.get('task', '')}")
  return "\n".join(lines)


def format_server_descriptions(
  tools_by_server: dict[str, list[dict[str, Any]]],
  server_index: dict[str, dict[str, Any]],
) -> str:
  server_descs = []

  for server_id in sorted(tools_by_server.keys()):
    server_tools = tools_by_server.get(server_id, [])
    tools_with_schema = [tool for tool in server_tools if tool.get("inputSchema") is not None]
    server_data = server_index.get(server_id, {})
    server_info = server_data.get("server", {})
    server_name = server_info.get("displayName", server_info.get("qualifiedName", "Unknown Server"))
    server_desc = server_info.get("description", "No description available")

    desc = f"### {server_name}\n"
    desc += f"**Description**: {server_desc}\n\n"
    desc += "**Available Tools**:\n"

    if not tools_with_schema:
      desc += "- (No tools with input schema available)\n"
      server_descs.append(desc)
      continue

    for i, tool in enumerate(tools_with_schema, 1):
      tool_name = tool.get("name", "Unknown Tool")
      tool_desc = str(tool.get("description", "No description available"))
      for marker in ["\nArgs:", "\nArguments:", "\nParameters:"]:
        if marker in tool_desc:
          tool_desc = tool_desc.split(marker, 1)[0].strip()
      if not tool_desc:
        tool_desc = "No description available"

      input_schema_json = json.dumps(tool.get("inputSchema"), indent=2, ensure_ascii=True)
      desc += f"{i}. **{tool_name}**: {tool_desc}\n"
      desc += "   **Input Schema**:\n"
      desc += f"```json\n{input_schema_json}\n```\n"

    server_descs.append(desc)

  return "\n".join(server_descs).strip()


def normalize_task_for_query(task_text: str) -> str:
  return re.sub(r"[.?!\s]+$", "", task_text.strip())


def build_default_reference_query(occupation_title: str, task_text: str) -> str:
  normalized_task = normalize_task_for_query(task_text)
  if normalized_task:
    return f"how do {occupation_title} {normalized_task}?"
  return f"how do {occupation_title} work?"


def build_task_references(
  occupation_title: str,
  onet_code: str,
  tasks: list[dict[str, Any]],
  task_refs_index: dict[tuple[str, str], dict[str, Any]] | None,
  no_refs: bool,
  ref_lookup_stats: TaskRefLookupStats,
) -> str:
  references = []
  refs_enabled = not no_refs and task_refs_index is not None

  for task in tasks:
    task_text = str(task.get("task", ""))
    query = build_default_reference_query(occupation_title, task_text)
    passage_text = "N/A"

    if refs_enabled:
      task_onet_code = str(task.get("onet_soc_code") or onet_code).strip()
      task_id = str(task.get("task_id", "")).strip()
      found = False
      has_non_empty_result = False

      if task_onet_code and task_id:
        ref_entry = task_refs_index.get((task_onet_code, task_id))
        if ref_entry is not None:
          found = True
          ref_query = str(ref_entry.get("query", "")).strip()
          if ref_query:
            query = ref_query

          results = ref_entry.get("results", [])
          if isinstance(results, list) and results:
            top_result = results[0]
            if isinstance(top_result, dict):
              top_passage = top_result.get("passage_text")
            else:
              top_passage = top_result
            if top_passage is not None and str(top_passage).strip():
              passage_text = str(top_passage)
              has_non_empty_result = True

      ref_lookup_stats.record(found=found, has_non_empty_result=has_non_empty_result)

    references.append(f"**Query**: {query}\n**Result**: {passage_text}")

  return "\n\n".join(references)


def build_server_context(
  selected_server_ids: list[str],
  server_index: dict[str, dict[str, Any]],
  mcp_servers_dir: str,
) -> ServerContext:
  seen_server_ids = sorted(set(selected_server_ids))
  tools_by_server: dict[str, list[dict[str, Any]]] = {}
  matched_servers_meta = []
  mcp_servers_metadata = []

  for server_id in seen_server_ids:
    server_data = server_index.get(server_id)
    if server_data is None:
      continue

    server_info = server_data.get("server", {})
    server_name = server_info.get("displayName", server_info.get("qualifiedName", ""))
    server_description = server_info.get("description", "")
    server_tools = server_info.get("tools", [])

    if server_tools:
      tools_by_server[server_id] = server_tools

    matched_servers_meta.append({"server_id": server_id, "server_name": server_name})
    mcp_servers_metadata.append(
      {
        "server_id": server_id,
        "server_name": server_name,
        "server_description": server_description,
        "tools": [
          {
            "name": tool.get("name"),
            "description": tool.get("description"),
            "inputSchema": tool.get("inputSchema"),
          }
          for tool in server_tools
        ],
        "source_file_path": os.path.join(mcp_servers_dir, f"{server_id}.json"),
      }
    )

  return ServerContext(tools_by_server, matched_servers_meta, mcp_servers_metadata)


def render_prompt(prompt_template: str, context: dict[str, Any]) -> str:
  prompt = prompt_template
  for key, value in context.items():
    prompt = prompt.replace(f"{{{key}}}", str(value))

  # Only treat placeholder-like tokens (starting with letter/underscore) as unresolved.
  # This avoids false positives for literal regex quantifiers in tool schemas like "{2}".
  unresolved = re.findall(r"\{[A-Z_][A-Z0-9_]*\}", prompt)
  if unresolved:
    placeholders = ", ".join(sorted(set(unresolved)))
    raise ValueError(f"Prompt template placeholders were not fully resolved: {placeholders}")

  return prompt


def build_prompt_record(
  i: int,
  combo_record: dict[str, Any],
  occupation_dict: dict[str, dict[str, str]],
  server_index: dict[str, dict[str, Any]],
  prompt_template: str,
  args: argparse.Namespace,
  args_dict: dict[str, Any],
  task_refs_index: dict[tuple[str, str], dict[str, Any]] | None,
  ref_lookup_stats: TaskRefLookupStats,
) -> dict[str, Any]:
  onet_code = combo_record["onet_code"]
  task_combo = combo_record["tasks"]
  selected_server_ids = list(combo_record["selected_server_ids"])

  occupation_info = occupation_dict.get(onet_code, {"title": "Unknown Occupation", "description": ""})
  occupation_title = occupation_info["title"]
  occupation_description = occupation_info["description"]

  server_context = build_server_context(selected_server_ids, server_index, args.mcp_servers_dir)

  prompt_context = {
    "NUM_TASKS": args.num_tasks,
    "NUM_TOOLS": args.num_tools,
    "OCCUPATION": occupation_title,
    "OCCUPATION_DESCRIPTION": occupation_description,
    "TASKS": format_tasks_list(task_combo),
    "SERVER_DESCRIPTIONS": format_server_descriptions(server_context.tools_by_server, server_index),
    "TASK_REFERENCES": build_task_references(
      occupation_title=occupation_title,
      onet_code=onet_code,
      tasks=task_combo,
      task_refs_index=task_refs_index,
      no_refs=args.no_refs,
      ref_lookup_stats=ref_lookup_stats,
    ),
  }
  prompt = render_prompt(prompt_template, prompt_context)

  return {
    "messages": [{"role": "user", "content": prompt}],
    "metadata": {
      "prompt_id": f"{i:08d}",
      "row_id": i,
      "mode": "onet_tasks",
      "question_gen_args": args_dict,
      "onet_soc_code": onet_code,
      "occupation_title": occupation_title,
      "tasks": [{"task_id": task.get("task_id"), "task": task.get("task")} for task in task_combo],
      "matched_servers": server_context.matched_servers_meta,
      "mcp_servers": server_context.mcp_servers_metadata,
    },
  }


def generate_and_write_prompts_streaming(
  total_prompts: int,
  worker_count: int,
  combo_iterator: Iterator[dict[str, Any]],
  build_row_fn: Callable[[int, dict[str, Any]], dict[str, Any]],
  output_file_path: str,
) -> int:
  max_in_flight = max(worker_count * 4, 1)
  next_submit_idx = 0
  next_write_idx = 0
  written_count = 0
  pending_results: dict[int, dict[str, Any]] = {}
  pbar = tqdm(total=total_prompts, desc="Generating prompts")
  combo_iter_exhausted = False

  with open(output_file_path, "w", encoding="utf-8") as f:
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
      future_to_idx: dict[Any, int] = {}

      def submit_next() -> None:
        nonlocal next_submit_idx, combo_iter_exhausted
        if next_submit_idx >= total_prompts or combo_iter_exhausted:
          return
        try:
          combo_record = next(combo_iterator)
        except StopIteration:
          combo_iter_exhausted = True
          return
        future = executor.submit(build_row_fn, next_submit_idx, combo_record)
        future_to_idx[future] = next_submit_idx
        next_submit_idx += 1

      for _ in range(min(max_in_flight, total_prompts)):
        submit_next()

      while future_to_idx:
        done, _ = wait(future_to_idx.keys(), return_when=FIRST_COMPLETED)

        for future in done:
          idx = future_to_idx.pop(future)
          try:
            result = future.result()
          except Exception as exc:
            pbar.close()
            raise RuntimeError(f"Failed generating prompt for row {idx}") from exc

          pending_results[idx] = result
          pbar.update(1)

          while next_write_idx in pending_results:
            next_result = pending_results.pop(next_write_idx)
            f.write(json.dumps(next_result) + "\n")
            next_write_idx += 1
            written_count += 1

          submit_next()

      if pending_results:
        pbar.close()
        raise RuntimeError("Prompt generation completed with unwritten pending results.")

      if next_submit_idx != total_prompts:
        pbar.close()
        raise RuntimeError(
          f"Prompt generation ended early: submitted {next_submit_idx} of {total_prompts} prompts"
        )

      if written_count != total_prompts:
        pbar.close()
        raise RuntimeError(
          f"Prompt generation completed with incomplete writes: wrote {written_count} of {total_prompts}"
        )

  pbar.close()
  return written_count


def load_inputs(paths: dict[str, str], args: argparse.Namespace) -> InputBundle:
  print(f"Loading tasks data from {paths['tasks_to_servers_path']}...")
  tasks_list = load_jsonl_to_list(paths["tasks_to_servers_path"])
  print(f"Loaded {len(tasks_list)} task records")

  print(f"Loading occupation data from {paths['occupation_data_path']}...")
  occupation_dict = load_occupation_data(paths["occupation_data_path"])

  print(f"Loading prompt template from {paths['prompt_template_path']}...")
  prompt_template = load_prompt_template(paths["prompt_template_path"])

  print(f"Loading MCP server metadata from {args.mcp_servers_dir}...")
  server_index = create_server_metadata_index(tasks_list, args.mcp_servers_dir)

  return InputBundle(tasks_list, occupation_dict, prompt_template, server_index)


def build_generation_pool(
  tasks_list: list[dict[str, Any]],
  server_index: dict[str, dict[str, Any]],
  num_tasks: int,
) -> GenerationPool:
  tasks_list_filtered = filter_tasks_with_loadable_servers(tasks_list, server_index)
  occupation_to_tasks = create_occupation_to_tasks_index(tasks_list_filtered)
  valid_onet_codes = get_valid_onet_codes(occupation_to_tasks, num_tasks)
  return GenerationPool(tasks_list_filtered, occupation_to_tasks, valid_onet_codes)


def main() -> None:
  args = parse_args()
  print("Tool Use Question Generation (O*NET task-first approach)")
  print(f"Arguments:\n{args}")

  if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)

  script_dir = os.path.dirname(os.path.abspath(__file__))
  paths = resolve_paths(script_dir, args.no_refs)
  inputs = load_inputs(paths, args)
  pool = build_generation_pool(inputs.tasks_list, inputs.server_index, args.num_tasks)

  if not pool.valid_onet_codes:
    raise ValueError(
      f"No occupations found with >= {args.num_tasks} tasks with matched, loadable servers. "
      "Try reducing --num_tasks."
    )

  combo_count = count_combo_records(
    occupation_to_tasks=pool.occupation_to_tasks,
    valid_onet_codes=pool.valid_onet_codes,
    num_tasks=args.num_tasks,
    limit=args.total_prompts,
  )
  total_prompts = compute_total_prompts(args.total_prompts, combo_count)
  total_prompts_tag = str(args.total_prompts) if args.total_prompts is not None else "all"
  output_paths = prepare_output_paths(args, total_prompts_tag)
  args_dict = save_generation_args(args, output_paths.args_file_path)

  serialize_combos_streaming(
    occupation_to_tasks=pool.occupation_to_tasks,
    valid_onet_codes=pool.valid_onet_codes,
    num_tasks=args.num_tasks,
    total_prompts=total_prompts,
    combos_parquet_path=output_paths.combos_parquet_path,
  )
  task_refs_index = None
  ref_lookup_stats = TaskRefLookupStats()

  if not args.no_refs:
    print(f"Loading task refs from {paths['task_refs_path']}...")
    task_refs_index = load_task_refs_index(paths["task_refs_path"])

  worker_count = init_generation_settings(args)

  def build_row_fn(i: int, combo_record: dict[str, Any]) -> dict[str, Any]:
    return build_prompt_record(
      i=i,
      combo_record=combo_record,
      occupation_dict=inputs.occupation_dict,
      server_index=inputs.server_index,
      prompt_template=inputs.prompt_template,
      args=args,
      args_dict=args_dict,
      task_refs_index=task_refs_index,
      ref_lookup_stats=ref_lookup_stats,
    )

  written_count = generate_and_write_prompts_streaming(
    total_prompts=total_prompts,
    worker_count=worker_count,
    combo_iterator=itertools.islice(
      iter_combo_records(pool.occupation_to_tasks, pool.valid_onet_codes, args.num_tasks),
      total_prompts,
    ),
    build_row_fn=build_row_fn,
    output_file_path=output_paths.output_file_path,
  )

  print(f"Finished. Total prompts: {written_count}")
  print(f"Total combinations available: {combo_count}")
  if not args.no_refs:
    print(
      "Task ref lookup summary: "
      f"requested={ref_lookup_stats.requested}, "
      f"hits={ref_lookup_stats.hits}, "
      f"empty_results={ref_lookup_stats.empty_results}, "
      f"misses={ref_lookup_stats.misses}"
    )
  print(f"Output file: {output_paths.output_file_path}")
  print(f"Combinations parquet: {output_paths.combos_parquet_path}")


if __name__ == "__main__":
  main()
