"""
Match MCP servers to O*NET workplace tasks using Qwen3-Embedding.

Uses instruction-aware embeddings with different instructions for queries (tasks)
vs documents (MCP servers) for improved semantic matching.
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
    List of dicts with onet_soc_code, task_id, task, task_type
  """
  raw_data = read_tsv(filepath)
  tasks = []
  for row in raw_data:
    tasks.append({
      'onet_soc_code': row['O*NET-SOC Code'],
      'task_id': row['Task ID'],
      'task': row['Task'],
      'task_type': row.get('Task Type', 'Unknown')
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


def load_mcp_servers(mcp_dir: str) -> List[Dict]:
  """
  Load all MCP server metadata from JSON files.

  Returns:
    List of dicts with server_name, overview, tools, labels, filename
  """
  servers = []
  mcp_path = Path(mcp_dir)
  json_files = sorted(mcp_path.glob("*.json"))

  print(f"Loading MCP servers from {mcp_dir}...")
  for json_file in tqdm(json_files, desc="Loading MCP servers"):
    try:
      with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

      # Extract relevant fields
      metadata = data.get('metadata', {})
      labels = data.get('labels', {})
      server_info = metadata.get('server_info_crawled', {})
      remote_response = metadata.get('remote_server_response', {})

      server_name = metadata.get('server_name', '') or server_info.get('name', '')
      overview = server_info.get('overview', '')

      # Get tools from remote_server_response or server_info_crawled
      tools = remote_response.get('tools', []) or server_info.get('tools', [])

      # Skip servers without valid tools
      if not labels.get('is_connected', False) or not tools:
        continue

      servers.append({
        'server_name': server_name,
        'overview': overview,
        'tools': tools,
        'primary_label': labels.get('primary_label', ''),
        'secondary_labels': labels.get('secondary_labels', []),
        'analysis': labels.get('analysis', ''),
        'filename': json_file.name
      })
    except Exception as e:
      print(f"Warning: Failed to load {json_file}: {e}")
      continue

  print(f"Loaded {len(servers)} valid MCP servers")
  return servers


def build_mcp_document_text(server: Dict) -> str:
  """
  Build document text for an MCP server.
  Combines server_name, overview, and tool descriptions.
  NO instruction prefix (document side for Qwen3-Embedding).

  Args:
    server: MCP server dict

  Returns:
    Combined document text
  """
  parts = []

  # Server name
  if server.get('server_name'):
    parts.append(f"Server: {server['server_name']}")

  # Overview
  if server.get('overview'):
    parts.append(f"Overview: {server['overview']}")

  # Tool descriptions (all tools, no truncation)
  tools = server.get('tools', [])
  if tools:
    tool_descs = []
    for i, tool in enumerate(tools, 1):
      tool_name = tool.get('name', '')
      tool_desc = tool.get('description', '')
      if tool_name and tool_desc:
        tool_descs.append(f"{i}. {tool_name}: {tool_desc}")
      elif tool_name:
        tool_descs.append(f"{i}. {tool_name}")

    if tool_descs:
      parts.append("Tools:\n" + "\n".join(tool_descs))

  return "\n".join(parts)


def format_task_as_query(task: str, occupation_title: Optional[str] = None) -> str:
  """
  Format a task as an instruction-aware query for Qwen3-Embedding.

  Args:
    task: The task description
    occupation_title: Optional occupation title for context

  Returns:
    Instruction-prefixed query string
  """
  instruction = (
    "Instruct: Given a workplace scenario, retrieve software tools and MCP servers that could help automate, assist with, or help complete this task.\n"
    "Query: "
  )

  if occupation_title:
    return f"{instruction} {occupation_title} {task.lower()}"
  return f"{instruction}{task}"


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
  """
  Extract the last token's hidden state for each sequence in the batch.
  This is the convention for Qwen3-Embedding models.

  Args:
    last_hidden_states: Shape (batch_size, seq_len, hidden_dim)
    attention_mask: Shape (batch_size, seq_len)

  Returns:
    Pooled embeddings of shape (batch_size, hidden_dim)
  """
  left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
  if left_padding:
    return last_hidden_states[:, -1]
  else:
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def encode_texts_batch(
  model: AutoModel,
  tokenizer: AutoTokenizer,
  texts: List[str],
  batch_size: int = 16,
  max_length: int = 8192,
  device: str = "cuda"
) -> np.ndarray:
  """
  Batch encode texts using Qwen3-Embedding with last_token_pool.

  Args:
    model: The Qwen3-Embedding model
    tokenizer: The tokenizer
    texts: List of texts to encode
    batch_size: Batch size for encoding
    max_length: Maximum sequence length
    device: Device to use

  Returns:
    Normalized embeddings of shape (num_texts, hidden_dim)
  """
  embds = []

  for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
    batch_texts = texts[i:i + batch_size]

    # Tokenize
    batch_dict = tokenizer(
      batch_texts,
      max_length=max_length,
      padding=True,
      truncation=True,
      return_tensors="pt"
    )
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

    # Forward pass
    with torch.no_grad():
      outputs = model(**batch_dict)
      _embds = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # Normalize
    _embds = torch.nn.functional.normalize(_embds, p=2, dim=1)
    embds.append(_embds.cpu().numpy())

  return np.vstack(embds)


def compute_similarities_and_match(
  task_embds: np.ndarray,
  server_embds: np.ndarray,
  tasks: List[Dict],
  servers: List[Dict],
  threshold: float = 0.5,
  use_occupation: bool = False
) -> List[Dict]:
  """
  Compute cosine similarities and match servers to tasks above threshold.

  Args:
    task_embds: Shape (num_tasks, hidden_dim)
    server_embds: Shape (num_servers, hidden_dim)
    tasks: List of task dicts
    servers: List of server dicts
    threshold: Minimum similarity score to include
    use_occupation: Whether to include occupation info in output

  Returns:
    List of server-centric result dicts with matched tasks for each server
  """
  print(f"Computing similarities (threshold={threshold})...")

  # Compute all pairwise similarities using matrix multiplication
  # Since embeddings are normalized, dot product = cosine similarity
  similarities = np.dot(task_embds, server_embds.T)  # (num_tasks, num_servers)

  results = []
  matched_task_ids = set()

  for j, server in enumerate(tqdm(servers, desc="Matching servers")):
    server_sims = similarities[:, j]

    # Find all tasks above threshold
    above_threshold = np.where(server_sims >= threshold)[0]

    # Sort by similarity (descending)
    sorted_indices = above_threshold[np.argsort(server_sims[above_threshold])[::-1]]

    matched_tasks = []
    for idx in sorted_indices:
      task = tasks[idx]
      task_entry = {
        'task_id': task['task_id'],
        'task': task['task'],
        'similarity_score': float(server_sims[idx])
      }
      if use_occupation:
        task_entry['onet_soc_code'] = task['onet_soc_code']
        task_entry['occupation_title'] = task.get('occupation_title', '')
      matched_tasks.append(task_entry)
      matched_task_ids.add(task['task_id'])

    results.append({
      'server_name': server['server_name'],
      'filename': server['filename'],
      'matched_tasks': matched_tasks
    })

  # Print statistics (before adding fallback matches)
  servers_with_matches = sum(1 for r in results if r['matched_tasks'])
  print(f"\n{'='*60}")
  print("Matching statistics")
  print(f"{'='*60}")
  print(f"Matched {len(matched_task_ids)} / {len(tasks)} tasks ({len(matched_task_ids) / len(tasks):.2%})")
  print(f"Matched {servers_with_matches} / {len(servers)} servers ({servers_with_matches / len(servers):.2%})")

  # For servers with no matches above threshold, add the closest task
  for j, result in enumerate(results):
    if not result['matched_tasks']:
      server_sims = similarities[:, j]
      best_idx = np.argmax(server_sims)
      task = tasks[best_idx]
      task_entry = {
        'task_id': task['task_id'],
        'task': task['task'],
        'similarity_score': float(server_sims[best_idx])
      }
      if use_occupation:
        task_entry['onet_soc_code'] = task['onet_soc_code']
        task_entry['occupation_title'] = task.get('occupation_title', '')
      result['matched_tasks'].append(task_entry)
    print(f"Matched remaining {len(servers) - servers_with_matches} / {len(servers)} servers to their closest tasks")

  return results


def save_task_embds_parquet(
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
    'query_text': query_texts,
    'embedding': [emb.tolist() for emb in embeddings]
  }
  # Include occupation_title if present
  if tasks and 'occupation_title' in tasks[0]:
    data['occupation_title'] = [t.get('occupation_title', '') for t in tasks]
  df = pd.DataFrame(data)
  df.to_parquet(filepath, index=False)
  print(f"Saved task embeddings to {filepath}")


def load_task_embds_parquet(filepath: str) -> Tuple[List[Dict], List[str], np.ndarray]:
  """Load task embeddings from parquet file."""
  print(f"Loading task embeddings from {filepath}")
  df = pd.read_parquet(filepath)
  has_occupation = 'occupation_title' in df.columns
  tasks = []
  for _, row in df.iterrows():
    task = {
      'onet_soc_code': row['onet_soc_code'],
      'task_id': row['task_id'],
      'task': row['task'],
      'task_type': row['task_type']
    }
    if has_occupation:
      task['occupation_title'] = row['occupation_title']
    tasks.append(task)
  query_texts = df['query_text'].tolist()
  embeddings = np.array(df['embedding'].tolist())
  print(f"Loaded {len(tasks)} task embeddings")
  return tasks, query_texts, embeddings


def save_server_embds_parquet(
  servers: List[Dict],
  document_texts: List[str],
  embeddings: np.ndarray,
  filepath: str
):
  """Save server embeddings with texts to parquet file."""
  df = pd.DataFrame({
    'server_name': [s['server_name'] for s in servers],
    'filename': [s['filename'] for s in servers],
    'primary_label': [s['primary_label'] for s in servers],
    'overview': [s['overview'] for s in servers],
    'document_text': document_texts,
    'embedding': [emb.tolist() for emb in embeddings]
  })
  df.to_parquet(filepath, index=False)
  print(f"Saved server embeddings to {filepath}")


def load_server_embds_parquet(filepath: str) -> Tuple[List[Dict], List[str], np.ndarray]:
  """Load server embeddings from parquet file."""
  print(f"Loading server embeddings from {filepath}")
  df = pd.read_parquet(filepath)
  has_primary_label = 'primary_label' in df.columns
  has_overview = 'overview' in df.columns
  servers = []
  for _, row in df.iterrows():
    server = {
      'server_name': row['server_name'],
      'filename': row['filename'],
      'tools': []  # Tools not stored in parquet, not needed for matching
    }
    if has_primary_label:
      server['primary_label'] = row['primary_label']
    if has_overview:
      server['overview'] = row['overview']
    servers.append(server)
  document_texts = df['document_text'].tolist()
  embeddings = np.array(df['embedding'].tolist())
  print(f"Loaded {len(servers)} server embeddings")
  return servers, document_texts, embeddings


def main():
  parser = argparse.ArgumentParser(
    description='Match O*NET tasks to MCP servers using Qwen3-Embedding'
  )
  parser.add_argument(
    '--task-file',
    default='onet_db_30_1_text/Task Statements.txt',
    help='Path to O*NET Task Statements TSV file'
  )
  parser.add_argument(
    '--mcp-dir',
    default='../mcp_servers',
    help='Path to directory containing MCP server JSON files'
  )
  parser.add_argument(
    '--model',
    default='Qwen/Qwen3-Embedding-8B',
    help='Qwen3-Embedding model to use. The script should not be used with embedding models that do not use the last token pooling strategy.'
  )
  parser.add_argument(
    '--threshold',
    type=float,
    default=0.5,
    help='Minimum cosine similarity threshold for matching'
  )
  parser.add_argument(
    '--output',
    default='servers_to_tasks.jsonl',
    help='Output JSONL file path'
  )
  parser.add_argument(
    '--use-occupation',
    action='store_true',
    help='Load occupation titles and include them in task queries and output'
  )
  parser.add_argument(
    '--occupation-file',
    default='onet_db_30_1_text/Occupation Data.txt',
    help='Path to O*NET Occupation Data TSV file'
  )
  parser.add_argument(
    '--task-embds',
    default=None,
    help='Task embeddings parquet file (load if exists, save if generated). Default: {model}_task_embds.parquet'
  )
  parser.add_argument(
    '--server-embds',
    default=None,
    help='Server embeddings parquet file (load if exists, save if generated). Default: {model}_server_embds.parquet'
  )
  parser.add_argument(
    '--batch-size',
    type=int,
    default=8,
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
  task_embds_file = args.task_embds or f"{model_short}_task_embds.parquet"
  server_embds_file = args.server_embds or f"{model_short}_server_embds.parquet"

  # Load data
  print(f"Loading O*NET tasks from {args.task_file}...")
  tasks = load_tasks(args.task_file)
  print(f"Loaded {len(tasks)} tasks")

  # Load occupation titles if requested
  occupations = {}
  if args.use_occupation:
    print(f"Loading O*NET occupations from {args.occupation_file}...")
    occupations = load_occupations(args.occupation_file)
    print(f"Loaded {len(occupations)} occupations")
    # Add occupation_title to each task
    for task in tasks:
      task['occupation_title'] = occupations.get(task['onet_soc_code'], '')

  servers = load_mcp_servers(args.mcp_dir)

  # Check if we can load cached embeddings
  load_task_cache = args.task_embds and os.path.exists(args.task_embds)
  load_server_cache = args.server_embds and os.path.exists(args.server_embds)
  need_model = not (load_task_cache and load_server_cache)

  # Only load model and tokenizer if needed
  model = None
  tokenizer = None
  device = None

  if need_model:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    print(f"\nLoading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model = model.to(device)
    model.eval()

    # Use DataParallel if multiple GPUs are available
    if device == "cuda" and torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)

  # Load or compute task embeddings
  if load_task_cache:
    cached_tasks, task_texts, task_embds = load_task_embds_parquet(args.task_embds)
    # Verify consistency
    cached_task_ids = [t['task_id'] for t in cached_tasks]
    current_task_ids = [t['task_id'] for t in tasks]
    if cached_task_ids != current_task_ids:
      print("Warning: Cached task IDs differ from current file. Recomputing...")
      load_task_cache = False

  if not load_task_cache:
    print("Building query texts for tasks...")
    if args.use_occupation:
      task_texts = [format_task_as_query(t['task'], t.get('occupation_title')) for t in tasks]
    else:
      task_texts = [format_task_as_query(t['task']) for t in tasks]

    print(f"\nEncoding {len(task_texts)} task queries...")
    task_embds = encode_texts_batch(
      model, tokenizer, task_texts,
      batch_size=args.batch_size,
      max_length=args.max_length,
      device=device
    )
    save_task_embds_parquet(tasks, task_texts, task_embds, task_embds_file)

  # Load or compute server embeddings
  if load_server_cache:
    cached_servers, server_texts, server_embds = load_server_embds_parquet(args.server_embds)
    # Verify consistency
    cached_filenames = [s['filename'] for s in cached_servers]
    current_filenames = [s['filename'] for s in servers]
    if cached_filenames != current_filenames:
      print("Warning: Cached server filenames differ from current directory. Recomputing...")
      load_server_cache = False
    else:
      # Use cached servers for matching (has overview, primary_label needed for results)
      servers = cached_servers

  if not load_server_cache:
    print("Building document texts for MCP servers...")
    server_texts = [build_mcp_document_text(s) for s in servers]

    print(f"\nEncoding {len(server_texts)} MCP server documents...")
    server_embds = encode_texts_batch(
      model, tokenizer, server_texts,
      batch_size=args.batch_size,
      max_length=args.max_length,
      device=device
    )
    save_server_embds_parquet(servers, server_texts, server_embds, server_embds_file)

  # Compute similarities and match
  results = compute_similarities_and_match(
    task_embds, server_embds,
    tasks, servers,
    threshold=args.threshold,
    use_occupation=args.use_occupation
  )

  # Save results
  print(f"\nSaving results to {args.output}...")
  save_dataset(results, args.output, convert_to_jsonl=True)
  print(f"Done! Saved {len(results)} server matches to {args.output}")

if __name__ == "__main__":
  main()
