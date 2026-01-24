"""
Match individual MCP tools to O*NET workplace tasks using Qwen3-Embedding.

Uses instruction-aware embeddings with different instructions for queries (tasks)
vs documents (MCP tools) for improved semantic matching.

Each tool is embedded independently and matched to tasks. Results are tool-centric,
with each tool as a top-level entry containing its metadata and matched tasks.
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


def load_mcp_servers(mcp_dir: str) -> Tuple[List[Dict], List[Dict]]:
  """
  Load all MCP server metadata from JSON files and extract individual tools.

  Returns:
    servers: List of server dicts with server_id, server_name, filename
    tools: Flattened list of tool dicts with:
        - tool_name, tool_description, input_schema
        - server_idx, server_name, server_filename, tool_idx
  """
  servers = []
  tools = []
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
      server_tools = remote_response.get('tools', []) or server_info.get('tools', [])

      # Skip servers without valid tools
      if not labels.get('is_connected', False) or not server_tools:
        continue

      server_idx = len(servers)
      servers.append({
        'server_id': server_idx,
        'server_name': server_name,
        'overview': overview,
        'primary_label': labels.get('primary_label', ''),
        'secondary_labels': labels.get('secondary_labels', []),
        'analysis': labels.get('analysis', ''),
        'filename': json_file.name,
        'num_tools': len(server_tools)
      })

      # Extract individual tools with server reference
      for tool_idx, tool in enumerate(server_tools):
        tools.append({
          'tool_idx': tool_idx,
          'tool_name': tool.get('name', ''),
          'tool_description': tool.get('description', ''),
          'input_schema': tool.get('inputSchema', {}),
          'server_idx': server_idx,
          'server_name': server_name,
          'server_analysis': labels.get('analysis', ''),
          'server_filename': json_file.name,
        })

    except Exception as e:
      print(f"Warning: Failed to load {json_file}: {e}")
      continue

  print(f"Loaded {len(servers)} valid MCP servers with {len(tools)} tools")
  return servers, tools


def build_tool_document_text(tool: Dict, include_server: bool = True) -> str:
  """
  Build document text for an individual MCP tool.
  Combines tool name, description, server name, server analysis, and parameter descriptions.
  NO instruction prefix (document side for Qwen3-Embedding).

  Args:
    tool: Tool dict with tool_name, tool_description, input_schema, server_name, server_analysis
    include_server: Whether to include server info for disambiguation

  Returns:
    Combined document text
  """
  parts = []

  # Tool name
  if tool.get('tool_name'):
    parts.append(f"Tool: {tool['tool_name']}")

  # Tool description
  if tool.get('tool_description'):
    parts.append(f"Tool Description: {tool['tool_description']}")

  if include_server:
    # Server name for context/disambiguation
    if tool.get('server_name'):
      parts.append(f"Server: {tool['server_name']}")
    # Server analysis/description for additional context
    if tool.get('server_analysis'):
      parts.append(f"Server Description: {tool['server_analysis']}")

  # Parameter descriptions from input_schema
  input_schema = tool.get('input_schema', {})
  properties = input_schema.get('properties', {})
  if properties:
    param_descs = []
    for param_name, param_info in properties.items():
      param_desc = param_info.get('description', '')
      if param_desc:
        param_descs.append(f"- {param_name}: {param_desc}")
      else:
        param_descs.append(f"- {param_name}")

    if param_descs:
      parts.append("Parameters:\n" + "\n".join(param_descs))

  return "\n".join(parts)


def format_task_as_query(task: str, occupation_title: str) -> str:
  """
  Format a task as an instruction-aware query for Qwen3-Embedding.

  Uses instruction-aware embedding format where the instruction describes
  the retrieval goal and the query contains the actual content to match.

  Args:
    task: The task description
    occupation_title: The occupation title

  Returns:
    Instruction-prefixed query string
  """
  instruction = (
    f"Instruct: Retrieve software tools that could help {occupation_title} "
    f"complete the following workplace task.\n"
    "Query: "
  )
  return f"{instruction}{task.lower()}"


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
  tool_embds: np.ndarray,
  tasks: List[Dict],
  tools: List[Dict],
  threshold: float = 0.5,
  min_matches: int = 0
) -> List[Dict]:
  """
  Compute cosine similarities and match tools to tasks above threshold.
  Returns a flat list where each tool is a top-level entry.

  Args:
    task_embds: Shape (num_tasks, hidden_dim)
    tool_embds: Shape (num_tools, hidden_dim)
    tasks: List of task dicts
    tools: List of tool dicts with server_idx references
    threshold: Minimum similarity score to include
    min_matches: Minimum number of tasks to match per tool. If 0, only matches
                 above threshold are returned. If > 0, ensures at least this many
                 tasks per tool by adding top-k closest if needed.

  Returns:
    List of tool-centric result dicts, each containing tool metadata and matched tasks
  """
  print(f"Computing similarities (threshold={threshold})...")

  # Compute all pairwise similarities using matrix multiplication
  # Since embeddings are normalized, dot product = cosine similarity
  similarities = np.dot(task_embds, tool_embds.T)  # (num_tasks, num_tools)

  # Build tool results first
  tool_results = []
  matched_task_ids = set()
  tools_with_matches = 0

  for j, tool in enumerate(tqdm(tools, desc="Matching tools")):
    tool_sims = similarities[:, j]

    # Find all tasks above threshold
    above_threshold = np.where(tool_sims >= threshold)[0]

    # Sort by similarity (descending)
    sorted_indices = above_threshold[np.argsort(tool_sims[above_threshold])[::-1]]

    matched_tasks = []
    for idx in sorted_indices:
      task = tasks[idx]
      task_entry = {
        'task_id': task['task_id'],
        'onet_soc_code': task['onet_soc_code'],
        'task': task['task'],
        'similarity_score': float(tool_sims[idx])
      }
      matched_tasks.append(task_entry)
      matched_task_ids.add(task['task_id'])

    # If we have fewer than min_matches, add top-k closest tasks
    if len(matched_tasks) < min_matches:
      # Get indices of top-k tasks sorted by similarity
      top_k_indices = np.argsort(tool_sims)[::-1][:min_matches]
      already_added = {tasks[idx]['task_id'] for idx in sorted_indices}

      for idx in top_k_indices:
        if len(matched_tasks) >= min_matches:
          break
        task = tasks[idx]
        if task['task_id'] in already_added:
          continue
        task_entry = {
          'task_id': task['task_id'],
          'onet_soc_code': task['onet_soc_code'],
          'task': task['task'],
          'similarity_score': float(tool_sims[idx])
        }
        matched_tasks.append(task_entry)
        matched_task_ids.add(task['task_id'])

    if len(above_threshold) > 0:
      tools_with_matches += 1

    tool_results.append({
      'tool_idx': tool['tool_idx'],
      'tool_name': tool['tool_name'],
      'tool_description': tool['tool_description'],
      'input_schema': tool.get('input_schema', {}),
      'server_idx': tool['server_idx'],
      'server_name': tool['server_name'],
      'server_analysis': tool.get('server_analysis', ''),
      'server_filename': tool['server_filename'],
      'matched_tasks': matched_tasks
    })

  # Print statistics
  print(f"\n{'='*60}")
  print("Matching statistics")
  print(f"{'='*60}")
  print(f"Matched {len(matched_task_ids)} / {len(tasks)} tasks ({len(matched_task_ids) / len(tasks):.2%})")
  print(f"Matched {tools_with_matches} / {len(tools)} tools above threshold ({tools_with_matches / len(tools):.2%})")
  if min_matches > 0:
    print(f"All {len(tools)} tools have at least {min_matches} match(es) (including fallback)")
  else:
    print(f"Only returning matches above threshold (min_matches=0)")

  return tool_results


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


def save_tool_embds_parquet(
  tools: List[Dict],
  document_texts: List[str],
  embeddings: np.ndarray,
  filepath: str
):
  """Save tool embeddings with texts to parquet file."""
  df = pd.DataFrame({
    'tool_idx': [t['tool_idx'] for t in tools],
    'tool_name': [t['tool_name'] for t in tools],
    'tool_description': [t['tool_description'] for t in tools],
    'server_idx': [t['server_idx'] for t in tools],
    'server_name': [t['server_name'] for t in tools],
    'server_analysis': [t.get('server_analysis', '') for t in tools],
    'server_filename': [t['server_filename'] for t in tools],
    'document_text': document_texts,
    'embedding': [emb.tolist() for emb in embeddings]
  })
  df.to_parquet(filepath, index=False)
  print(f"Saved tool embeddings to {filepath}")


def load_tool_embds_parquet(filepath: str) -> Tuple[List[Dict], List[str], np.ndarray]:
  """Load tool embeddings from parquet file."""
  print(f"Loading tool embeddings from {filepath}")
  df = pd.read_parquet(filepath)
  tools = []
  for _, row in df.iterrows():
    tool = {
      'tool_idx': row['tool_idx'],
      'tool_name': row['tool_name'],
      'tool_description': row['tool_description'],
      'server_idx': row['server_idx'],
      'server_analysis': row['server_analysis'],
      'server_name': row['server_name'],
      'server_filename': row['server_filename'],
      'input_schema': {}  # Not stored in parquet, not needed for matching
    }
    tools.append(tool)
  document_texts = df['document_text'].tolist()
  embeddings = np.array(df['embedding'].tolist())
  print(f"Loaded {len(tools)} tool embeddings")
  return tools, document_texts, embeddings


def main():
  parser = argparse.ArgumentParser(
    description='Match O*NET tasks to MCP tools using Qwen3-Embedding'
  )
  parser.add_argument(
    '--mcp-dir',
    default='../mcp_servers',
    help='Path to directory containing MCP server JSON files'
  )
  parser.add_argument(
    '--tool-embds',
    default=None,
    help='Tool embeddings parquet file (load if exists, save if generated). Default: {model}_tool_embds.parquet'
  )
  parser.add_argument(
    '--task-file',
    default='onet_db_30_1_text/Task Statements.txt',
    help='Path to O*NET Task Statements TSV file'
  )
  parser.add_argument(
    '--task-embds',
    default=None,
    help='Task embeddings parquet file (load if exists, save if generated). Default: {model}_task_embds.parquet'
  )
  parser.add_argument(
    '--occupation-file',
    default='onet_db_30_1_text/Occupation Data.txt',
    help='Path to O*NET Occupation Data TSV file'
  )
  parser.add_argument(
    '--threshold',
    type=float,
    default=0.5,
    help='Minimum cosine similarity threshold for matching'
  )
  parser.add_argument(
    '--min-matches',
    type=int,
    default=0,
    help='Minimum number of tasks to match per tool. If 0, only matches above threshold are returned.'
  )
  parser.add_argument(
    '--output-file',
    default='tools_to_tasks.jsonl',
    help='Output JSONL file path'
  )
  parser.add_argument(
    '--model',
    default='Qwen/Qwen3-Embedding-8B',
    help='Qwen3-Embedding model to use. The script should not be used with embedding models that do not use the last token pooling strategy.'
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
  task_embds_file = args.task_embds or f"{model_short}_task_embds.parquet"
  tool_embds_file = args.tool_embds or f"{model_short}_tool_embds.parquet"

  # Load data
  print(f"Loading O*NET tasks from {args.task_file}...")
  tasks = load_tasks(args.task_file)
  print(f"Loaded {len(tasks)} tasks")

  # Load occupation titles
  print(f"Loading O*NET occupations from {args.occupation_file}...")
  occupations = load_occupations(args.occupation_file)
  print(f"Loaded {len(occupations)} occupations")
  for task in tasks:
    task['occupation_title'] = occupations.get(task['onet_soc_code'], '')

  servers, tools = load_mcp_servers(args.mcp_dir)

  # Check if we can load cached embeddings
  load_task_cache = args.task_embds and os.path.exists(args.task_embds)
  load_tool_cache = args.tool_embds and os.path.exists(args.tool_embds)
  need_model = not (load_task_cache and load_tool_cache)

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
    task_texts = [format_task_as_query(t['task'], t['occupation_title']) for t in tasks]

    print(f"\nEncoding {len(task_texts)} task queries...")
    task_embds = encode_texts_batch(
      model, tokenizer, task_texts,
      batch_size=args.batch_size,
      max_length=args.max_length,
      device=device
    )
    save_task_embds_parquet(tasks, task_texts, task_embds, task_embds_file)

  # Load or compute tool embeddings
  if load_tool_cache:
    cached_tools, tool_texts, tool_embds = load_tool_embds_parquet(args.tool_embds)
    # Verify consistency by checking tool count
    if len(cached_tools) != len(tools):
      print(f"Warning: Cached tools ({len(cached_tools)}) differ from current ({len(tools)}). Recomputing...")
      load_tool_cache = False
    else:
      # Use cached tools for matching
      tools = cached_tools

  if not load_tool_cache:
    print("Building document texts for MCP tools...")
    tool_texts = [build_tool_document_text(t) for t in tools]

    print(f"\nEncoding {len(tool_texts)} MCP tool documents...")
    tool_embds = encode_texts_batch(
      model, tokenizer, tool_texts,
      batch_size=args.batch_size,
      max_length=args.max_length,
      device=device
    )
    save_tool_embds_parquet(tools, tool_texts, tool_embds, tool_embds_file)

  # Compute similarities and match
  results = compute_similarities_and_match(
    task_embds, tool_embds,
    tasks, tools,
    threshold=args.threshold,
    min_matches=args.min_matches
  )

  # Save results
  print(f"\nSaving results to {args.output_file}...")
  save_dataset(results, args.output_file, convert_to_jsonl=True)

  # Print summary
  print(f"Done! Saved {len(results)} tools to {args.output_file}")

if __name__ == "__main__":
  main()
