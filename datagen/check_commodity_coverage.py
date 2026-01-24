"""
Check MCP tool coverage against O*NET Technology Skills commodity titles.

Uses Qwen3-Embedding for semantic matching between commodity titles
and individual MCP tool descriptions.
"""

import os
import csv
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer

from utils import save_dataset


def read_tsv(filepath: str) -> List[Dict]:
  """Read a tab-separated file and return list of dicts."""
  with open(filepath, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    return list(reader)


def load_commodity_titles(
  tech_skills_file: str,
  unspsc_file: str = None
) -> Tuple[List[str], List[str], List[str]]:
  """
  Load unique commodity titles, codes, and definitions.

  Args:
    tech_skills_file: Path to Technology Skills.txt
    unspsc_file: Optional path to UNSPSC CSV file with commodity definitions

  Returns:
    Tuple of:
      - List of unique commodity titles (sorted alphabetically)
      - List of commodity codes (in same order as titles)
      - List of commodity definitions (in same order as titles, empty string if not found)
  """
  # Load definitions from UNSPSC file if provided
  definitions: Dict[str, str] = {}
  if unspsc_file and os.path.exists(unspsc_file):
    with open(unspsc_file, 'r', encoding='utf-8') as f:
      reader = csv.DictReader(f)
      for row in reader:
        code = row.get('Commodity', '').strip()
        definition = row.get('Commodity Definition', '').strip()
        if code and definition:
          definitions[code] = definition
    print(f"Loaded {len(definitions)} commodity definitions from UNSPSC file")

  # Load commodity titles and codes from Technology Skills file
  raw_data = read_tsv(tech_skills_file)

  # Collect codes for each commodity title
  commodity_codes: Dict[str, str] = {}
  for row in raw_data:
    title = row.get('Commodity Title', '').strip()
    code = row.get('Commodity Code', '').strip()
    if title and code:
      commodity_codes[title] = code

  # Get unique titles sorted alphabetically
  unique_titles = sorted(commodity_codes.keys())
  # Get codes in same order as titles
  codes = [commodity_codes[title] for title in unique_titles]
  # Get definitions in same order (empty string if not available)
  defs = [definitions.get(commodity_codes[title], '') for title in unique_titles]

  if definitions:
    num_with_defs = sum(1 for d in defs if d)
    print(f"Matched {num_with_defs}/{len(unique_titles)} commodities with definitions")

  print(f"Loaded {len(unique_titles)} unique commodity titles")
  return unique_titles, codes, defs


def format_commodity_as_query(title: str, definition: str) -> str:
  """
  Format a commodity title as an instruction-aware query for Qwen3-Embedding.

  Args:
    title: The commodity title (e.g., "Document management software")
    definition: commodity definition from UNSPSC

  Returns:
    Instruction-prefixed query string
  """
  return (
    f"Instruct: Retrieve software tools that provide similar functionality as software in the following category.\n"
    f"Query: {definition}"
  )

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
          'tool_name': tool.get('name', ''),
          'tool_description': tool.get('description', ''),
          'input_schema': tool.get('inputSchema', {}),
          'server_idx': server_idx,
          'server_name': server_name,
          'server_analysis': labels.get('analysis', ''),
          'server_filename': json_file.name,
          'tool_idx': tool_idx
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
  commodity_codes: List[str],
  commodity_defs: List[str],
  query_texts: List[str],
  embeddings: np.ndarray,
  filepath: str
):
  """Save commodity embeddings with codes, definitions, and texts to parquet file."""
  df = pd.DataFrame({
    'commodity_code': commodity_codes,
    'commodity_title': commodity_titles,
    'commodity_definition': commodity_defs,
    'query_text': query_texts,
    'embedding': [emb.tolist() for emb in embeddings]
  })
  df.to_parquet(filepath, index=False)
  print(f"Saved commodity embeddings to {filepath}")


def load_commodity_embds_parquet(filepath: str) -> Tuple[List[str], List[str], List[str], List[str], np.ndarray]:
  """Load commodity embeddings from parquet file."""
  print(f"Loading commodity embeddings from {filepath}")
  df = pd.read_parquet(filepath)
  commodity_codes = df['commodity_code'].tolist()
  commodity_titles = df['commodity_title'].tolist()
  # Handle older parquet files without definitions
  if 'commodity_definition' in df.columns:
    commodity_defs = df['commodity_definition'].tolist()
  else:
    commodity_defs = [''] * len(commodity_titles)
  query_texts = df['query_text'].tolist()
  embeddings = np.array(df['embedding'].tolist())
  return commodity_titles, commodity_codes, commodity_defs, query_texts, embeddings


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


def compute_commodity_coverage(
  commodity_embds: np.ndarray,
  tool_embds: np.ndarray,
  commodity_titles: List[str],
  commodity_codes: List[str],
  commodity_defs: List[str],
  tools: List[Dict],
  threshold: float = 0.5
) -> Tuple[List[Dict], Dict]:
  """
  Match commodities to tools and compute coverage statistics.

  Args:
    commodity_embds: Shape (num_commodities, hidden_dim)
    tool_embds: Shape (num_tools, hidden_dim)
    commodity_titles: List of commodity titles
    commodity_codes: List of commodity codes (same order as titles)
    commodity_defs: List of commodity definitions (same order as titles)
    tools: List of tool dicts
    threshold: Minimum similarity score to include

  Returns:
    Tuple of (results list, stats dict)
  """
  print(f"Computing similarities (threshold={threshold})...")

  # Compute all pairwise similarities using matrix multiplication
  # Since embeddings are normalized, dot product = cosine similarity
  similarities = np.dot(commodity_embds, tool_embds.T)  # (num_commodities, num_tools)

  results = []
  covered_commodities = set()
  matched_tool_ids = set()  # Track unique tools by (server_filename, tool_idx)

  for i, (title, code, definition) in enumerate(tqdm(
      zip(commodity_titles, commodity_codes, commodity_defs),
      desc="Matching commodities", total=len(commodity_titles)
  )):
    sims = similarities[i]

    # Find all tools above threshold
    above_threshold = np.where(sims >= threshold)[0]

    # Sort by similarity (descending)
    sorted_indices = above_threshold[np.argsort(sims[above_threshold])[::-1]]

    matched_tools = []
    for idx in sorted_indices:
      tool = tools[idx]
      matched_tools.append({
        'tool_idx': tool['tool_idx'],
        'tool_name': tool['tool_name'],
        'tool_description': tool['tool_description'],
        'server_idx': tool['server_idx'],
        'server_name': tool['server_name'],
        'server_analysis': tool.get('server_analysis', ''),
        'server_filename': tool['server_filename'],
        'similarity_score': float(sims[idx])
      })
      matched_tool_ids.add((tool['server_filename'], tool['tool_idx']))

    if matched_tools:
      covered_commodities.add(title)

    results.append({
      'commodity_code': code,
      'commodity_title': title,
      'commodity_definition': definition,
      'matched_tools': matched_tools
    })

  # Compute statistics
  uncovered = [title for title in commodity_titles if title not in covered_commodities]
  stats = {
    'total_commodities': len(commodity_titles),
    'covered_commodities': len(covered_commodities),
    'coverage_rate': len(covered_commodities) / len(commodity_titles) if commodity_titles else 0,
    'total_tools': len(tools),
    'matched_tools': len(matched_tool_ids),
    'utilization_rate': len(matched_tool_ids) / len(tools) if tools else 0,
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
  print(f"Tools: {stats['matched_tools']} / {stats['total_tools']} matched "
        f"({stats['utilization_rate']:.1%})")

  if stats['uncovered_commodities']:
    print(f"\n{'='*60}")
    print(f"Uncovered Commodity Titles ({len(stats['uncovered_commodities'])})")
    print(f"{'='*60}")
    for title in sorted(stats['uncovered_commodities']):
      print(f"  - {title}")


def main():
  parser = argparse.ArgumentParser(
    description='Check MCP tool coverage against O*NET commodity titles using Qwen3-Embedding'
  )
  parser.add_argument(
    '--mcp-dir',
    default='../mcp_servers',
    help='Path to directory containing MCP server JSON files'
  )
  parser.add_argument(
    '--tool-embds',
    default=None,
    help='Tool embeddings parquet file (load if exists, save if generated)'
  )
  parser.add_argument(
    '--tech-skills-file',
    default='onet_db_30_1_text/Technology Skills.txt',
    help='Path to O*NET Technology Skills TSV file'
  )
  parser.add_argument(
    '--unspsc-file',
    default='onet_db_30_1_text/unspsc-english-v260801.1.csv',
    help='Path to UNSPSC CSV file with commodity definitions'
  )
  parser.add_argument(
    '--commodity-embds',
    default=None,
    help='Commodity embeddings parquet file (load if exists, save if generated)'
  )
  parser.add_argument(
    '--threshold',
    type=float,
    default=0.5,
    help='Minimum cosine similarity threshold for matching'
  )
  parser.add_argument(
    '--output-file',
    default='commodities_to_tools.jsonl',
    help='Output JSONL file path for detailed results'
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
  commodity_embds_file = args.commodity_embds or f"{model_short}_commodity_embds.parquet"
  tool_embds_file = args.tool_embds or f"{model_short}_tool_embds.parquet"

  # Load commodity titles, codes, and definitions
  print(f"Loading commodity data from {args.tech_skills_file}...")
  commodity_titles, commodity_codes, commodity_defs = load_commodity_titles(
    args.tech_skills_file, args.unspsc_file
  )

  # Load MCP servers and tools
  servers, tools = load_mcp_servers(args.mcp_dir)

  # Check if we can load cached embeddings
  load_commodity_cache = args.commodity_embds and os.path.exists(args.commodity_embds)
  load_tool_cache = args.tool_embds and os.path.exists(args.tool_embds)
  need_model = not (load_commodity_cache and load_tool_cache)

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
    cached_titles, cached_codes, cached_defs, query_texts, commodity_embds = load_commodity_embds_parquet(args.commodity_embds)
    # Verify consistency (check titles and definitions)
    if cached_titles != commodity_titles or cached_defs != commodity_defs:
      if cached_titles != commodity_titles:
        print("Warning: Cached commodity titles differ from current file. Recomputing...")
      else:
        print("Warning: Cached commodity definitions differ from current file. Recomputing...")
      load_commodity_cache = False
    else:
      # Use cached codes and defs
      commodity_codes = cached_codes
      commodity_defs = cached_defs

  if not load_commodity_cache:
    print("Building query texts for commodities...")
    query_texts = [
      format_commodity_as_query(title, definition)
      for title, definition in zip(commodity_titles, commodity_defs)
    ]

    print(f"\nEncoding {len(query_texts)} commodity queries...")
    commodity_embds = encode_texts_batch(
      model, tokenizer, query_texts,
      batch_size=args.batch_size,
      max_length=args.max_length,
      device=device
    )
    save_commodity_embds_parquet(
      commodity_titles, commodity_codes, commodity_defs, query_texts,
      commodity_embds, commodity_embds_file
    )

  # Load or compute tool embeddings
  if load_tool_cache:
    cached_tools, document_texts, tool_embds = load_tool_embds_parquet(args.tool_embds)
    # Verify consistency by checking tool count
    if len(cached_tools) != len(tools):
      print(f"Warning: Cached tools ({len(cached_tools)}) differ from current ({len(tools)}). Recomputing...")
      load_tool_cache = False
    else:
      # Use cached tools for matching
      tools = cached_tools

  if not load_tool_cache:
    print("Building document texts for MCP tools...")
    document_texts = [build_tool_document_text(t) for t in tools]

    print(f"\nEncoding {len(document_texts)} MCP tool documents...")
    tool_embds = encode_texts_batch(
      model, tokenizer, document_texts,
      batch_size=args.batch_size,
      max_length=args.max_length,
      device=device
    )
    save_tool_embds_parquet(tools, document_texts, tool_embds, tool_embds_file)

  # Verify embeddings are normalized
  print("\nVerifying embeddings are normalized...")
  commodity_norms = np.linalg.norm(commodity_embds, axis=1)
  tool_norms = np.linalg.norm(tool_embds, axis=1)
  print(f"Commodity embedding norms - mean: {commodity_norms.mean():.4f}, std: {commodity_norms.std():.6f}")
  print(f"Tool embedding norms - mean: {tool_norms.mean():.4f}, std: {tool_norms.std():.6f}")

  # Compute coverage
  results, stats = compute_commodity_coverage(
    commodity_embds, tool_embds,
    commodity_titles, commodity_codes, commodity_defs, tools,
    threshold=args.threshold
  )

  # Print coverage report
  print_coverage_report(stats)

  # Save detailed results
  print(f"\nSaving results to {args.output_file}...")
  save_dataset(results, args.output_file, convert_to_jsonl=True)
  print(f"Done! Saved {len(results)} commodity results to {args.output_file}")

if __name__ == "__main__":
  main()
