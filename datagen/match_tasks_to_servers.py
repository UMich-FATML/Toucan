"""
Match O*NET workplace tasks to Smithery MCP servers using Qwen3-Embedding.

Uses instruction-aware embeddings with different instructions for queries (tasks)
vs documents (MCP servers) for improved semantic matching.

Each task is matched to whole servers (not individual tools). Results are task-centric,
with each task as a top-level entry containing its metadata and matched servers.
"""

import os
import csv
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer

from utils import save_dataset


def read_tsv(filepath: str) -> List[Dict]:
  """Read a tab-separated file and return list of dicts."""
  with open(filepath, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    return list(reader)


def load_tasks(filepath: str) -> List[Dict]:
  """
  Load O*NET tasks from Task Statements.txt.

  Returns:
    List of dicts with onet_soc_code, task_id, task, task_type, emerging
  """
  raw_data = read_tsv(filepath)
  tasks = []
  for row in raw_data:
    tasks.append({
      'onet_soc_code': row['O*NET-SOC Code'],
      'task_id': row['Task ID'],
      'task': row['Task'],
      'task_type': row.get('Task Type', 'Unknown'),
      'emerging': False
    })
  return tasks


def load_emerging_tasks(filepath: str) -> List[Dict]:
  """
  Load O*NET emerging tasks from Emerging Tasks.txt.

  This file has no Task ID column, so we generate synthetic IDs (E_0, E_1, ...).

  Returns:
    List of dicts with onet_soc_code, task_id, task, task_type, emerging
  """
  raw_data = read_tsv(filepath)
  tasks = []
  for i, row in enumerate(raw_data):
    tasks.append({
      'onet_soc_code': row['O*NET-SOC Code'],
      'task_id': f'E_{i}',
      'task': row['Task'],
      'task_type': row.get('Category', 'Emerging'),
      'emerging': True
    })
  return tasks


def load_occupations(filepath: str) -> Dict[str, str]:
  """
  Load O*NET occupation titles from Occupation Data.txt.

  Returns:
    Dict mapping onet_soc_code -> occupation_title
  """
  raw_data = read_tsv(filepath)
  occupations = {}
  for row in raw_data:
    occupations[row['O*NET-SOC Code']] = row['Title']
  return occupations


def format_task_as_query(task: str, occupation_title: str) -> str:
  """
  Format a task as an instruction-aware query for Qwen3-Embedding.

  Args:
    task: The task description
    occupation_title: The occupation title

  Returns:
    Instruction-prefixed query string
  """
  instruction = (
    f"Instruct: Retrieve software tools that can help {occupation_title} "
    f"complete the following task.\n"
    "Query: "
  )
  return f"{instruction}{task.lower()}"


def format_server_as_document(json_path: Path) -> Optional[Tuple[Dict, str]]:
  """
  Load a Smithery MCP server JSON file and format its metadata as document text.

  Args:
    json_path: Path to the MCP server JSON file

  Returns:
    Tuple of (server_metadata_dict, document_text) or None if server is invalid
  """
  try:
    with open(json_path, 'r', encoding='utf-8') as f:
      data = json.load(f)
  except Exception as e:
    print(f"Warning: Failed to load {json_path}: {e}")
    return None

  server = data.get('server') or {}

  # Skip servers with no tools
  server_tools = server.get('tools') or []
  if not server_tools:
    return None

  # Skip servers with no analysis
  analysis = data.get('analysis', '')
  if not analysis:
    return None

  server_name = server.get('displayName', '')
  description = server.get('description', '')
  labels = data.get('labels') or []
  categories = data.get('categories') or []

  # Build document text
  parts = []
  if server_name:
    parts.append(f"Server: {server_name}")
  if description:
    parts.append(f"Description: {description}")
  if analysis:
    parts.append(f"Analysis: {analysis}")
  if labels:
    parts.append(f"Categories: {', '.join(labels)}")
  if categories:
    parts.append(f"Tags: {', '.join(categories)}")

  # Add tools (name + description only, no parameter details)
  tool_lines = []
  for tool in server_tools:
    tool_name = tool.get('name', '')
    tool_desc = tool.get('description', '')
    if tool_name and tool_desc:
      tool_lines.append(f"- {tool_name}: {tool_desc}")
    elif tool_name:
      tool_lines.append(f"- {tool_name}")

  if tool_lines:
    parts.append("Tools:\n" + "\n".join(tool_lines))

  document_text = "\n".join(parts)

  server_meta = {
    'server_id': server.get('id', ''),
    'server_name': server_name,
    'filename': json_path.name,
    'num_tools': len(server_tools)
  }

  return server_meta, document_text


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
  """
  Extract the last token's hidden state for each sequence in the batch.
  This is the convention for Qwen3-Embedding models.
  """
  left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
  if left_padding:
    return last_hidden_states[:, -1]
  else:
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def batch_encode_texts(
  model: AutoModel,
  tokenizer: AutoTokenizer,
  texts: List[str],
  batch_size: int = 16,
  max_length: int = 8192,
  device: str = "cuda"
) -> np.ndarray:
  """
  Batch encode texts using Qwen3-Embedding with last_token_pool.

  Returns:
    Normalized embeddings of shape (num_texts, hidden_dim)
  """
  embds = []

  for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
    batch_texts = texts[i:i + batch_size]

    batch_dict = tokenizer(
      batch_texts,
      max_length=max_length,
      padding=True,
      truncation=True,
      return_tensors="pt"
    )
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

    with torch.no_grad():
      outputs = model(**batch_dict)
      _embds = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    _embds = torch.nn.functional.normalize(_embds, p=2, dim=1)
    embds.append(_embds.cpu().numpy())

  return np.vstack(embds)


def save_task_embeds_parquet(
  tasks: List[Dict],
  query_texts: List[str],
  embeddings: np.ndarray,
  filepath: str
):
  """Save task embeddings with texts to parquet file."""
  data = {
    'onet_soc_code': [t['onet_soc_code'] for t in tasks],
    'task_id': [t['task_id'] for t in tasks],
    'task': [t['task'] for t in tasks],
    'task_type': [t['task_type'] for t in tasks],
    'emerging': [t['emerging'] for t in tasks],
    'occupation_title': [t.get('occupation_title', '') for t in tasks],
    'query_text': query_texts,
    'embedding': [emb.tolist() for emb in embeddings]
  }
  df = pd.DataFrame(data)
  df.to_parquet(filepath, index=False)
  print(f"Saved task embeddings to {filepath}")


def load_task_embeds_parquet(filepath: str) -> Tuple[List[Dict], List[str], np.ndarray]:
  """Load task embeddings from parquet file."""
  print(f"Loading task embeddings from {filepath}")
  df = pd.read_parquet(filepath)
  tasks = []
  for _, row in df.iterrows():
    task = {
      'onet_soc_code': row['onet_soc_code'],
      'task_id': row['task_id'],
      'task': row['task'],
      'task_type': row['task_type'],
      'emerging': row.get('emerging', False),
      'occupation_title': row.get('occupation_title', '')
    }
    tasks.append(task)
  query_texts = df['query_text'].tolist()
  embeddings = np.array(df['embedding'].tolist())
  print(f"Loaded {len(tasks)} task embeddings")
  return tasks, query_texts, embeddings


def save_server_embeds_parquet(
  servers: List[Dict],
  document_texts: List[str],
  embeddings: np.ndarray,
  filepath: str
):
  """Save server embeddings with lightweight cache data to parquet file."""
  df = pd.DataFrame({
    'server_id': [s['server_id'] for s in servers],
    'server_name': [s['server_name'] for s in servers],
    'filename': [s['filename'] for s in servers],
    'document_text': document_texts,
    'embedding': [emb.tolist() for emb in embeddings]
  })
  df.to_parquet(filepath, index=False)
  print(f"Saved server embeddings to {filepath}")


def load_server_embeds_parquet(filepath: str) -> Tuple[List[Dict], List[str], np.ndarray]:
  """Load server embeddings from parquet file.

  Returns:
    Tuple of (servers, document_texts, embeddings)
  """
  print(f"Loading server embeddings from {filepath}")
  df = pd.read_parquet(filepath)
  servers = []
  for _, row in df.iterrows():
    servers.append({
      'server_id': row['server_id'],
      'server_name': row['server_name'],
      'filename': row['filename'],
    })
  document_texts = df['document_text'].tolist()
  embeddings = np.array(df['embedding'].tolist())
  print(f"Loaded {len(servers)} server embeddings")
  return servers, document_texts, embeddings


def compute_task_server_coverage(
  task_embeds: np.ndarray,
  server_embeds: np.ndarray,
  tasks: List[Dict],
  servers: List[Dict],
  threshold: float = 0.5
) -> Tuple[List[Dict], Dict]:
  """
  Match tasks to servers and compute coverage statistics.

  Args:
    task_embeds: Shape (num_tasks, hidden_dim)
    server_embeds: Shape (num_servers, hidden_dim)
    tasks: List of task dicts
    servers: List of server metadata dicts
    threshold: Minimum similarity score to include

  Returns:
    Tuple of (results list, stats dict)
  """
  print(f"Computing similarities (threshold={threshold})...")

  # Cosine similarity via dot product (embeddings are normalized)
  similarities = np.dot(task_embeds, server_embeds.T)  # (num_tasks, num_servers)

  results = []
  covered_tasks = 0
  matched_server_idxs = set()

  for i, task in enumerate(tqdm(tasks, desc="Matching tasks")):
    task_sims = similarities[i]

    # Find all servers above threshold
    above_threshold = np.where(task_sims >= threshold)[0]

    # Sort by similarity (descending)
    sorted_indices = above_threshold[np.argsort(task_sims[above_threshold])[::-1]]

    matched_servers = []
    for idx in sorted_indices:
      server = servers[idx]
      matched_servers.append({
        'server_id': server['server_id'],
        'server_name': server['server_name'],
        'similarity_score': round(float(task_sims[idx]), 4)
      })
      matched_server_idxs.add(idx)

    if matched_servers:
      covered_tasks += 1

    results.append({
      'onet_soc_code': task['onet_soc_code'],
      'task_id': task['task_id'],
      'occupation_title': task.get('occupation_title', ''),
      'task': task['task'],
      'emerging': task['emerging'],
      'matched_servers': matched_servers
    })

  stats = {
    'total_tasks': len(tasks),
    'covered_tasks': covered_tasks,
    'coverage_rate': covered_tasks / len(tasks) if tasks else 0,
    'total_servers': len(servers),
    'matched_servers': len(matched_server_idxs),
    'utilization_rate': len(matched_server_idxs) / len(servers) if servers else 0,
  }

  return results, stats


def print_coverage_report(stats: Dict):
  """Print coverage summary statistics to stdout."""
  print(f"\n{'='*60}")
  print("Coverage Summary")
  print(f"{'='*60}")
  print(f"Tasks: {stats['covered_tasks']} / {stats['total_tasks']} covered "
        f"({stats['coverage_rate']:.1%})")
  print(f"Servers: {stats['matched_servers']} / {stats['total_servers']} matched "
        f"({stats['utilization_rate']:.1%})")


def main():
  parser = argparse.ArgumentParser(
    description='Match O*NET tasks to Smithery MCP servers using Qwen3-Embedding'
  )
  parser.add_argument(
    '--mcp-dir',
    default='../mcp_servers/smithery_mcp_servers_0210',
    help='Path to directory containing Smithery MCP server JSON files'
  )
  parser.add_argument(
    '--task-file',
    default='onet_db_30_1_text/Task Statements.txt',
    help='Path to O*NET Task Statements TSV file'
  )
  parser.add_argument(
    '--emerging-file',
    default='onet_db_30_1_text/Emerging Tasks.txt',
    help='Path to O*NET Emerging Tasks TSV file'
  )
  parser.add_argument(
    '--occupation-file',
    default='onet_db_30_1_text/Occupation Data.txt',
    help='Path to O*NET Occupation Data TSV file'
  )
  parser.add_argument(
    '--task-embeds',
    default=None,
    help='Task embeddings parquet file (load if exists, save if generated)'
  )
  parser.add_argument(
    '--server-embeds',
    default=None,
    help='Server embeddings parquet file (load if exists, save if generated)'
  )
  parser.add_argument(
    '--threshold',
    type=float,
    default=0.5,
    help='Minimum cosine similarity threshold for matching'
  )
  parser.add_argument(
    '--output-file',
    default='tasks_to_smithery_servers.jsonl',
    help='Output JSONL file path'
  )
  parser.add_argument(
    '--model',
    default='Qwen/Qwen3-Embedding-8B',
    help='Qwen3-Embedding model to use'
  )
  parser.add_argument(
    '--batch-size',
    type=int,
    default=16,
    help='Batch size for encoding'
  )
  parser.add_argument(
    '--max-length',
    type=int,
    default=8192,
    help='Maximum sequence length for tokenization'
  )

  args = parser.parse_args()

  # Default embedding file paths
  model_short = args.model.split('/')[-1]
  task_embeds_file = args.task_embeds or f"{model_short}_task_server_task_embeds.parquet"
  server_embeds_file = args.server_embeds or f"{model_short}_task_server_server_embeds.parquet"

  # Load tasks from both sources
  print(f"Loading O*NET tasks from {args.task_file}...")
  tasks = load_tasks(args.task_file)
  print(f"Loaded {len(tasks)} task statements")

  print(f"Loading O*NET emerging tasks from {args.emerging_file}...")
  emerging_tasks = load_emerging_tasks(args.emerging_file)
  print(f"Loaded {len(emerging_tasks)} emerging tasks")

  tasks = tasks + emerging_tasks
  print(f"Total tasks: {len(tasks)}")

  # Load occupation titles
  print(f"Loading O*NET occupations from {args.occupation_file}...")
  occupations = load_occupations(args.occupation_file)
  print(f"Loaded {len(occupations)} occupations")
  for task in tasks:
    task['occupation_title'] = occupations.get(task['onet_soc_code'], '')

  # Load MCP servers
  mcp_path = Path(args.mcp_dir)
  json_files = sorted(mcp_path.glob("*.json"))
  print(f"Loading MCP servers from {args.mcp_dir}...")
  servers = []
  document_texts = []
  for json_file in tqdm(json_files, desc="Loading MCP servers"):
    result = format_server_as_document(json_file)
    if result is not None:
      server_meta, doc_text = result
      servers.append(server_meta)
      document_texts.append(doc_text)
  print(f"Loaded {len(servers)} valid MCP servers")

  # Check if we can load cached embeddings
  load_task_cache = args.task_embeds and os.path.exists(args.task_embeds)
  load_server_cache = args.server_embeds and os.path.exists(args.server_embeds)
  need_model = not (load_task_cache and load_server_cache)

  # Only load model and tokenizer if needed
  model = None
  tokenizer = None
  device = None

  if need_model:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model = model.to(device)
    model.eval()

    if device == "cuda" and torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)

  # Load or compute task embeddings
  if load_task_cache:
    cached_tasks, task_texts, task_embeds = load_task_embeds_parquet(args.task_embeds)
    cached_task_ids = [t['task_id'] for t in cached_tasks]
    current_task_ids = [t['task_id'] for t in tasks]
    if cached_task_ids != current_task_ids:
      print("Warning: Cached task IDs differ from current file. Recomputing...")
      load_task_cache = False

  if not load_task_cache:
    print("Building query texts for tasks...")
    task_texts = [format_task_as_query(t['task'], t['occupation_title']) for t in tasks]

    print(f"\nEncoding {len(task_texts)} task queries...")
    task_embeds = batch_encode_texts(
      model, tokenizer, task_texts,
      batch_size=args.batch_size,
      max_length=args.max_length,
      device=device
    )
    save_task_embeds_parquet(tasks, task_texts, task_embeds, task_embeds_file)

  # Load or compute server embeddings
  if load_server_cache:
    cached_servers, cached_doc_texts, server_embeds = load_server_embeds_parquet(args.server_embeds)
    cached_filenames = [s['filename'] for s in cached_servers]
    current_filenames = [s['filename'] for s in servers]
    if cached_filenames != current_filenames:
      print("Warning: Cached server filenames differ from current. Recomputing...")
      load_server_cache = False
    else:
      document_texts = cached_doc_texts
      servers = cached_servers

  if not load_server_cache:
    print(f"\nEncoding {len(document_texts)} MCP server documents...")
    server_embeds = batch_encode_texts(
      model, tokenizer, document_texts,
      batch_size=args.batch_size,
      max_length=args.max_length,
      device=device
    )
    save_server_embeds_parquet(servers, document_texts, server_embeds, server_embeds_file)

  # Compute coverage
  results, stats = compute_task_server_coverage(
    task_embeds, server_embeds,
    tasks, servers,
    threshold=args.threshold
  )

  # Print coverage report
  print_coverage_report(stats)

  # Save results
  print(f"\nSaving results to {args.output_file}...")
  save_dataset(results, args.output_file, convert_to_jsonl=True)
  print(f"Done! Saved {len(results)} task results to {args.output_file}")


if __name__ == "__main__":
  main()
