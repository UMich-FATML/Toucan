"""
Check MCP server coverage against O*NET Technology Skills commodity titles.

Uses Qwen3-Embedding for semantic matching between commodity titles
and MCP server descriptions.
"""

import os
import csv
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer

from utils import save_dataset


def read_tsv(filepath: str) -> List[Dict]:
  """Read a tab-separated file and return list of dicts."""
  with open(filepath, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    return list(reader)


def load_commodity_titles(filepath: str) -> Tuple[List[str], Dict[str, List[str]]]:
  """
  Load unique commodity titles and their example software from Technology Skills.txt.

  Returns:
    Tuple of:
      - List of unique commodity titles
      - Dict mapping commodity title to list of example software names
  """
  raw_data = read_tsv(filepath)

  # Collect examples for each commodity title
  commodity_examples: Dict[str, List[str]] = defaultdict(list)
  for row in raw_data:
    title = row.get('Commodity Title', '').strip()
    example = row.get('Example', '').strip()
    if title and example:
      # Avoid duplicates
      if example not in commodity_examples[title]:
        commodity_examples[title].append(example)

  # Get unique titles sorted alphabetically
  unique_titles = sorted(commodity_examples.keys())

  print(f"Loaded {len(unique_titles)} unique commodity titles")
  return unique_titles, dict(commodity_examples)


def format_commodity_as_query(
  title: str,
  examples: Optional[List[str]] = None,
  use_examples: bool = False,
  max_examples: int = 5
) -> str:
  """
  Format a commodity title as an instruction-aware query for Qwen3-Embedding.

  Args:
    title: The commodity title (e.g., "Document management software")
    examples: Optional list of example software names
    use_examples: Whether to include example software names in the query
    max_examples: Maximum number of examples to include

  Returns:
    Instruction-prefixed query string
  """
  instruction = (
    "Instruct: Given a software category from the O*NET Technology Skills database, "
    "retrieve MCP servers and tools that provide functionality in this software category.\n"
    "Query: "
  )

  if use_examples and examples:
    # Select up to max_examples examples
    selected_examples = examples[:max_examples]
    examples_str = ", ".join(selected_examples)
    return f"{instruction}{title} (examples: {examples_str})"
  return f"{instruction}{title}"


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
  batch_size: int = 8,
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


def save_commodity_embds_parquet(
  commodity_titles: List[str],
  query_texts: List[str],
  embeddings: np.ndarray,
  filepath: str
):
  """Save commodity embeddings with texts to parquet file."""
  df = pd.DataFrame({
    'commodity_title': commodity_titles,
    'query_text': query_texts,
    'embedding': [emb.tolist() for emb in embeddings]
  })
  df.to_parquet(filepath, index=False)
  print(f"Saved commodity embeddings to {filepath}")


def load_commodity_embds_parquet(filepath: str) -> Tuple[List[str], List[str], np.ndarray]:
  """Load commodity embeddings from parquet file."""
  print(f"Loading commodity embeddings from {filepath}")
  df = pd.read_parquet(filepath)
  commodity_titles = df['commodity_title'].tolist()
  query_texts = df['query_text'].tolist()
  embeddings = np.array(df['embedding'].tolist())
  return commodity_titles, query_texts, embeddings


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
    'document_text': document_texts,
    'embedding': [emb.tolist() for emb in embeddings]
  })
  df.to_parquet(filepath, index=False)
  print(f"Saved server embeddings to {filepath}")


def load_server_embds_parquet(filepath: str) -> Tuple[List[str], List[str], List[str], np.ndarray]:
  """Load server embeddings from parquet file."""
  print(f"Loading server embeddings from {filepath}")
  df = pd.read_parquet(filepath)
  server_names = df['server_name'].tolist()
  filenames = df['filename'].tolist()
  document_texts = df['document_text'].tolist()
  embeddings = np.array(df['embedding'].tolist())
  return server_names, filenames, document_texts, embeddings


def compute_commodity_coverage(
  commodity_embds: np.ndarray,
  server_embds: np.ndarray,
  commodity_titles: List[str],
  servers: List[Dict],
  threshold: float = 0.5
) -> Tuple[List[Dict], Dict]:
  """
  Match commodities to servers and compute coverage statistics.

  Args:
    commodity_embds: Shape (num_commodities, hidden_dim)
    server_embds: Shape (num_servers, hidden_dim)
    commodity_titles: List of commodity titles
    servers: List of server dicts
    threshold: Minimum similarity score to include

  Returns:
    Tuple of (results list, stats dict)
  """
  print(f"Computing similarities (threshold={threshold})...")

  # Compute all pairwise similarities using matrix multiplication
  # Since embeddings are normalized, dot product = cosine similarity
  similarities = np.dot(commodity_embds, server_embds.T)  # (num_commodities, num_servers)

  results = []
  covered_commodities = set()
  matched_server_filenames = set()

  for i, title in enumerate(tqdm(commodity_titles, desc="Matching commodities")):
    sims = similarities[i]

    # Find all servers above threshold
    above_threshold = np.where(sims >= threshold)[0]

    # Sort by similarity (descending)
    sorted_indices = above_threshold[np.argsort(sims[above_threshold])[::-1]]

    matched_servers = []
    for idx in sorted_indices:
      server = servers[idx]
      matched_servers.append({
        'server_name': server['server_name'],
        'filename': server['filename'],
        'similarity_score': float(sims[idx]),
        'primary_label': server['primary_label']
      })
      matched_server_filenames.add(server['filename'])

    if matched_servers:
      covered_commodities.add(title)

    results.append({
      'commodity_title': title,
      'matched_servers': matched_servers
    })

  # Compute statistics
  uncovered = [title for title in commodity_titles if title not in covered_commodities]
  stats = {
    'total_commodities': len(commodity_titles),
    'covered_commodities': len(covered_commodities),
    'coverage_rate': len(covered_commodities) / len(commodity_titles) if commodity_titles else 0,
    'total_servers': len(servers),
    'matched_servers': len(matched_server_filenames),
    'utilization_rate': len(matched_server_filenames) / len(servers) if servers else 0,
    'uncovered_commodities': uncovered
  }

  return results, stats


def print_coverage_report(stats: Dict):
  """Print coverage summary statistics to stdout."""
  print(f"\n{'='*60}")
  print("Coverage Summary")
  print(f"{'='*60}")
  print(f"Commodities: {stats['covered_commodities']} / {stats['total_commodities']} covered "
        f"({stats['coverage_rate']:.1%})")
  print(f"Servers: {stats['matched_servers']} / {stats['total_servers']} matched "
        f"({stats['utilization_rate']:.1%})")

  if stats['uncovered_commodities']:
    print(f"\n{'='*60}")
    print(f"Uncovered Commodity Titles ({len(stats['uncovered_commodities'])})")
    print(f"{'='*60}")
    for title in sorted(stats['uncovered_commodities']):
      print(f"  - {title}")


def main():
  parser = argparse.ArgumentParser(
    description='Check MCP server coverage against O*NET commodity titles using Qwen3-Embedding'
  )
  parser.add_argument(
    '--tech-skills-file',
    default='onet_db_30_1_text/Technology Skills.txt',
    help='Path to O*NET Technology Skills TSV file'
  )
  parser.add_argument(
    '--mcp-dir',
    default='../mcp_servers',
    help='Path to directory containing MCP server JSON files'
  )
  parser.add_argument(
    '--model',
    default='Qwen/Qwen3-Embedding-8B',
    help='Qwen3-Embedding model to use'
  )
  parser.add_argument(
    '--threshold',
    type=float,
    default=0.5,
    help='Minimum cosine similarity threshold for matching'
  )
  parser.add_argument(
    '--output',
    default='commodities_to_servers.jsonl',
    help='Output JSONL file path for detailed results'
  )
  parser.add_argument(
    '--commodity-embds',
    default=None,
    help='Commodity embeddings parquet file (load if exists, save if generated)'
  )
  parser.add_argument(
    '--server-embds',
    default=None,
    help='Server embeddings parquet file (load if exists, save if generated)'
  )
  parser.add_argument(
    '--use-examples',
    action='store_true',
    help='Include example software names in commodity queries'
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
  commodity_embds_file = args.commodity_embds or f"{model_short}_commodity_embds.parquet"
  server_embds_file = args.server_embds or f"{model_short}_server_embds.parquet"

  # Load commodity titles
  print(f"Loading commodity titles from {args.tech_skills_file}...")
  commodity_titles, commodity_examples = load_commodity_titles(args.tech_skills_file)

  # Load MCP servers
  servers = load_mcp_servers(args.mcp_dir)

  # Check if we can load cached embeddings
  load_commodity_cache = args.commodity_embds and os.path.exists(args.commodity_embds)
  load_server_cache = args.server_embds and os.path.exists(args.server_embds)
  need_model = not (load_commodity_cache and load_server_cache)

  # Model and tokenizer (only load if needed)
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

    # Use DataParallel if multiple GPUs are available
    if device == "cuda" and torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)

  # Load or compute commodity embeddings
  if load_commodity_cache:
    cached_titles, query_texts, commodity_embds = load_commodity_embds_parquet(args.commodity_embds)
    # Verify consistency
    if cached_titles != commodity_titles:
      print("Warning: Cached commodity titles differ from current file. Recomputing...")
      load_commodity_cache = False

  if not load_commodity_cache:
    print("Building query texts for commodities...")
    query_texts = [
      format_commodity_as_query(
        title,
        commodity_examples.get(title, []),
        use_examples=args.use_examples
      )
      for title in commodity_titles
    ]

    print(f"\nEncoding {len(query_texts)} commodity queries...")
    commodity_embds = encode_texts_batch(
      model, tokenizer, query_texts,
      batch_size=args.batch_size,
      max_length=args.max_length,
      device=device
    )
    save_commodity_embds_parquet(commodity_titles, query_texts, commodity_embds, commodity_embds_file)

  # Load or compute server embeddings
  if load_server_cache:
    cached_names, cached_filenames, document_texts, server_embds = load_server_embds_parquet(args.server_embds)
    # Verify consistency
    current_filenames = [s['filename'] for s in servers]
    if cached_filenames != current_filenames:
      print("Warning: Cached server filenames differ from current directory. Recomputing...")
      load_server_cache = False

  if not load_server_cache:
    print("Building document texts for MCP servers...")
    document_texts = [build_mcp_document_text(s) for s in servers]

    print(f"\nEncoding {len(document_texts)} MCP server documents...")
    server_embds = encode_texts_batch(
      model, tokenizer, document_texts,
      batch_size=args.batch_size,
      max_length=args.max_length,
      device=device
    )
    save_server_embds_parquet(servers, document_texts, server_embds, server_embds_file)

  # Verify embeddings are normalized
  print("\nVerifying embeddings are normalized...")
  commodity_norms = np.linalg.norm(commodity_embds, axis=1)
  server_norms = np.linalg.norm(server_embds, axis=1)
  print(f"Commodity embedding norms - mean: {commodity_norms.mean():.4f}, std: {commodity_norms.std():.6f}")
  print(f"Server embedding norms - mean: {server_norms.mean():.4f}, std: {server_norms.std():.6f}")

  # Compute coverage
  results, stats = compute_commodity_coverage(
    commodity_embds, server_embds,
    commodity_titles, servers,
    threshold=args.threshold
  )

  # Print coverage report
  print_coverage_report(stats)

  # Save detailed results
  print(f"\nSaving results to {args.output}...")
  save_dataset(results, args.output, convert_to_jsonl=True)
  print(f"Done! Saved {len(results)} commodity results to {args.output}")

if __name__ == "__main__":
  main()
