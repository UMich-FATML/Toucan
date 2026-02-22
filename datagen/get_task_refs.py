import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm


def get_args():
  parser = argparse.ArgumentParser(
    description="Collect search references for O*NET tasks."
  )
  parser.add_argument(
    "--task_file",
    type=str,
    default="onet_db_30_1_text/Task Statements.txt",
    help="Path to O*NET Task Statements file.",
  )
  parser.add_argument(
    "--occupation_file",
    type=str,
    default="onet_db_30_1_text/Occupation Data.txt",
    help="Path to O*NET Occupation Data file.",
  )
  parser.add_argument(
    "--search_pool_file",
    type=str,
    default="search_api_pool.json",
    help="Path to search endpoint pool JSON.",
  )
  parser.add_argument(
    "--output_file",
    type=str,
    default="task_refs.parquet",
    help="Output parquet file path.",
  )
  parser.add_argument("--k", type=int, default=10, help="Top-k search results to request.")
  parser.add_argument(
    "--num_workers",
    type=int,
    default=None,
    help="Concurrent workers. Default: number of search endpoints.",
  )
  parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Optional limit for number of task rows to process.",
  )
  parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume from existing output file by skipping completed tasks.",
  )
  return parser.parse_args()


def load_search_endpoints(search_pool_file):
  with open(search_pool_file, "r", encoding="utf-8") as f:
    data = json.load(f)

  api_pool = data.get("api_pool")
  if not isinstance(api_pool, list):
    raise ValueError(f"Invalid format in {search_pool_file}: expected list at 'api_pool'")

  endpoints = []
  for entry in api_pool:
    if isinstance(entry, dict):
      url = entry.get("url")
      if isinstance(url, str) and url.strip():
        endpoints.append(url.strip())

  if not endpoints:
    raise ValueError(f"No valid endpoint URLs found in {search_pool_file}")

  return endpoints


def load_joined_task_rows(task_file, occupation_file):
  task_df = pd.read_csv(task_file, sep="\t", dtype=str)
  occ_df = pd.read_csv(occupation_file, sep="\t", dtype=str)

  required_task_cols = ["O*NET-SOC Code", "Task ID", "Task"]
  required_occ_cols = ["O*NET-SOC Code", "Title"]

  for col in required_task_cols:
    if col not in task_df.columns:
      raise ValueError(f"Missing required column '{col}' in {task_file}")
  for col in required_occ_cols:
    if col not in occ_df.columns:
      raise ValueError(f"Missing required column '{col}' in {occupation_file}")

  merged = task_df.merge(
    occ_df[["O*NET-SOC Code", "Title"]],
    on="O*NET-SOC Code",
    how="left",
  )
  missing_titles = merged["Title"].isna().sum()
  if missing_titles > 0:
    raise ValueError(f"Found {missing_titles} task rows without matched occupation title")

  records = []
  for _, row in merged.iterrows():
    onet_soc_code = str(row["O*NET-SOC Code"]).strip()
    task_id = str(row["Task ID"]).strip()
    occupation_title = str(row["Title"]).strip()
    task_text = str(row["Task"]).strip()
    if not onet_soc_code or not task_id or not occupation_title or not task_text:
      continue
    records.append(
      {
        "onet_soc_code": onet_soc_code,
        "task_id": task_id,
        "occupation_title": occupation_title,
        "task": task_text,
      }
    )

  return records


def normalize_task_for_query(task_text):
  # Remove trailing punctuation before appending '?' so queries are consistent.
  stripped = re.sub(r"[.?!\s]+$", "", task_text.strip())
  return stripped


def build_query(occupation_title, task_text):
  normalized_task = normalize_task_for_query(task_text)
  return f"example of {occupation_title} {normalized_task[:-1]}"


def normalize_results_value(results):
  if isinstance(results, list):
    return results
  if hasattr(results, "tolist"):
    converted = results.tolist()
    if isinstance(converted, list):
      return converted
  return []


def load_existing_output_for_resume(output_file):
  if not os.path.exists(output_file):
    return set(), []

  existing_rows = []
  if output_file.lower().endswith(".parquet"):
    existing_df = pd.read_parquet(output_file)
    existing_rows = existing_df.to_dict(orient="records")
  else:
    with open(output_file, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          row = json.loads(line)
        except json.JSONDecodeError:
          continue
        existing_rows.append(row)

  seen = set()
  for row in existing_rows:
    row["results"] = normalize_results_value(row.get("results"))
    row["query"] = str(row.get("query", "")).strip()
    onet_soc_code = str(row.get("onet_soc_code", "")).strip()
    task_id = str(row.get("task_id", "")).strip()
    if onet_soc_code and task_id:
      seen.add((onet_soc_code, task_id))
  return seen, existing_rows


def fetch_results(query, endpoint, k):
  payload = {
    "query": query,
    "k": k,
    "rerank": True,
    "return_fulltext": False,
  }
  MAX_RETRIES = 3
  TIMEOUT_SEC = 30

  for attempt in range(1, MAX_RETRIES + 1):
    try:
      response = requests.post(endpoint, json=payload, timeout=TIMEOUT_SEC)
      response.raise_for_status()
      result_body = response.json()

      # Extract results from response dict or bare list.
      if isinstance(result_body, dict):
        results = result_body.get("results", [])
      else:
        results = []

      if isinstance(results, list) and len(results) > 0:
        return results[:k]
    except (requests.RequestException, ValueError):
      pass

    if attempt < MAX_RETRIES:
      time.sleep(attempt)

  return []


def process_row(row, endpoint, args):
  query = build_query(row["occupation_title"], row["task"])
  results = fetch_results(
    query=query,
    endpoint=endpoint,
    k=args.k,
  )

  # Keep output schema minimal and fixed.
  return {
    "onet_soc_code": row["onet_soc_code"],
    "task_id": row["task_id"],
    "query": query,
    "results": results,
  }


def main():
  args = get_args()

  if args.k <= 0:
    raise ValueError("--k must be a positive integer.")

  endpoints = load_search_endpoints(args.search_pool_file)
  all_rows = load_joined_task_rows(args.task_file, args.occupation_file)

  if args.limit is not None:
    all_rows = all_rows[: args.limit]

  existing_rows = []
  if args.resume:
    seen_keys, existing_rows = load_existing_output_for_resume(args.output_file)
    all_rows = [
      row
      for row in all_rows
      if (row["onet_soc_code"], row["task_id"]) not in seen_keys
    ]

  if not all_rows:
    print("No rows to process.")
    return

  num_workers = args.num_workers or len(endpoints)
  num_workers = max(1, num_workers)

  os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

  print(f"Total tasks to process: {len(all_rows)}")
  print(f"Using {num_workers} workers across {len(endpoints)} endpoint(s)")
  print(f"Output file: {args.output_file}")

  start_time = time.time()
  success_count = 0
  new_rows = []

  with ThreadPoolExecutor(max_workers=num_workers) as executor:
    future_to_meta = {}
    for idx, row in enumerate(all_rows):
      endpoint = endpoints[idx % len(endpoints)]
      future = executor.submit(process_row, row, endpoint, args)
      future_to_meta[future] = row

    with tqdm(total=len(future_to_meta), desc="Fetching task refs") as pbar:
      for future in as_completed(future_to_meta):
        row_meta = future_to_meta[future]
        try:
          output_row = future.result()
        except Exception:
          output_row = {
            "onet_soc_code": row_meta["onet_soc_code"],
            "task_id": row_meta["task_id"],
            "query": build_query(row_meta["occupation_title"], row_meta["task"]),
            "results": [],
          }
        output_row["results"] = normalize_results_value(output_row.get("results"))
        if len(output_row["results"]) == args.k:
          success_count += 1
        new_rows.append(output_row)
        pbar.update(1)

  output_rows = existing_rows + new_rows if args.resume else new_rows
  output_df = pd.DataFrame(output_rows, columns=["onet_soc_code", "task_id", "query", "results"])
  output_df.to_parquet(args.output_file, index=False)

  elapsed = time.time() - start_time
  print(f"Done. Success rows (exact k): {success_count}/{len(all_rows)}")
  print(f"Saved {len(output_rows)} rows to {args.output_file}")
  print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
  main()
