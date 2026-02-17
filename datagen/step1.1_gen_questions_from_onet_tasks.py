import os
import argparse
import json
import time
import random
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from utils import load_jsonl_to_list

################
# Use Cases
################
"""
This script generates questions for tool use scenarios using a TASK-FIRST approach via O*NET tasks.

Unlike the tool-first script (step1.1_gen_onet_questions_from_tools.py) which picks tools then invents
a scenario, this script starts from O*NET workplace tasks and selects tools as means to accomplish them.
The prompt is enriched with Knowledge domains and Skills from O*NET to ground scenarios more realistically.

The script uses DETERMINISTIC combo generation:
1. Loads tasks with matched MCP servers from tasks_to_smithery_servers.jsonl
2. Groups tasks by occupation, filters to tasks with matched servers
3. Enumerates all possible (occupation, task_combination) pairs
4. Orders by occupation code first, then task combinations within each occupation
5. Generates prompts for the first N combinations (where N = total_prompts)
6. Saves the full combinations dataframe as a parquet file

This ensures reproducibility - running with the same arguments always produces identical results.

Example Usage:

1. Basic usage - generate 100 prompts with 2 tasks each:
   python step1.1_gen_questions_from_onet_tasks.py --num_tasks 2 --total_prompts 100

2. Custom output folder and job name:
   python step1.1_gen_questions_from_onet_tasks.py --num_tasks 2 --total_prompts 1000 --output_folder ../data --job_name my_experiment

Key Parameters:
- --num_tasks: Number of O*NET tasks to include in each prompt (required)
- --total_prompts: Total number of prompts to generate (required)
- --top_k_knowledge: Number of top knowledge domains to include (default 5)
- --top_k_skills: Number of top skills to include (default 5)
- --seed: Random seed (only affects np.random, not combo generation which is deterministic)

Output Files:
- <output_dir>/<job_folder>/onet_tasks_*_prepared.jsonl: Generated prompts
- <output_dir>/<job_folder>/combos.parquet: All possible combinations
- <output_dir>/<job_folder>/generation_args.json: Arguments used for generation
"""

################
# Configurations
################
def get_args():
  parser = argparse.ArgumentParser(description="Tool Use Question Generation using O*NET Task-first Approach.")

  # Required parameters
  parser.add_argument("--num_tasks", type=int, required=True, help="Number of O*NET tasks per prompt.")
  parser.add_argument("--total_prompts", type=int, required=True, help="Total number of prompts to generate.")

  # Optional parameters
  parser.add_argument("--output_folder", type=str, default="../data", help="Output folder path.")
  parser.add_argument("--job_name", type=str, default=None, help="Job name for organization.")
  parser.add_argument("--timestamp", type=int, default=int(time.time()), help="Timestamp for the job.")
  parser.add_argument("--seed", type=int, default=None, help="Random seed.")
  parser.add_argument("--top_k_knowledge", type=int, default=5, help="Number of top knowledge domains to include.")
  parser.add_argument("--top_k_skills", type=int, default=5, help="Number of top skills to include.")
  parser.add_argument("--mcp_servers_dir", type=str, default="../mcp_servers/smithery_mcp_servers_0210", help="Directory containing MCP server JSON files.")

  return parser.parse_args()


################
# Data Loading Functions
################
def load_occupation_data(occupation_file_path):
  """
  Load occupation data from O*NET TSV file.

  Returns:
    dict: Mapping from onet_soc_code to {'title': ..., 'description': ...}
  """
  occupations = {}

  with open(occupation_file_path, 'r', encoding='utf-8') as f:
    # Skip header
    f.readline()

    for line in f:
      parts = line.strip().split('\t')
      if len(parts) >= 3:
        code = parts[0]
        title = parts[1]
        description = parts[2]
        occupations[code] = {
          'title': title,
          'description': description
        }

  print(f"Loaded {len(occupations)} occupations from {occupation_file_path}")
  return occupations


def load_onet_attribute(file_path, top_k):
  """
  Load an O*NET attribute file (Knowledge.txt or Skills.txt) and return
  top-K items by importance for each occupation.

  Filters to Scale ID == "IM" (Importance), groups by O*NET-SOC Code,
  sorts by Data Value descending, takes top-K.

  Args:
    file_path: Path to the O*NET TSV file
    top_k: Number of top items to return per occupation

  Returns:
    dict: Mapping from onet_soc_code to list of {'name': ..., 'importance': ...}
  """
  df = pd.read_csv(file_path, sep='\t')
  # Filter to Importance scale
  df = df[df['Scale ID'] == 'IM']
  # Sort by Data Value descending within each occupation
  df = df.sort_values(['O*NET-SOC Code', 'Data Value'], ascending=[True, False])
  # Take top-K per occupation
  result = {}
  for code, group in df.groupby('O*NET-SOC Code'):
    top_items = group.head(top_k)
    result[code] = [
      {'name': row['Element Name'], 'importance': row['Data Value']}
      for _, row in top_items.iterrows()
    ]

  print(f"Loaded attributes for {len(result)} occupations from {file_path} (top {top_k})")
  return result


def load_prompt_template(template_path):
  """Load the prompt template from file."""
  with open(template_path, 'r', encoding='utf-8') as f:
    return f.read()


################
# Index Creation Functions
################
def create_occupation_to_tasks_index(tasks_list):
  """
  Create an index from O*NET SOC codes to tasks that have matched servers.

  Args:
    tasks_list: List of task records from tasks_to_smithery_servers.jsonl

  Returns:
    dict: Mapping from onet_soc_code to list of task records (only those with matched servers)
  """
  occupation_to_tasks = defaultdict(list)

  for task in tasks_list:
    matched_servers = task.get('matched_servers', [])
    if matched_servers:  # Only include tasks with matched servers
      onet_code = task.get('onet_soc_code')
      if onet_code:
        occupation_to_tasks[onet_code].append(task)

  print(f"Built index with {len(occupation_to_tasks)} occupations (tasks with matched servers)")
  return occupation_to_tasks


def create_server_metadata_index(tasks_list, mcp_servers_dir):
  """
  Build a server_id -> server metadata index by loading MCP server JSON files.

  Args:
    tasks_list: List of task records (to extract unique server_ids)
    mcp_servers_dir: Directory containing MCP server JSON files

  Returns:
    dict: Mapping from server_id to server metadata dict
  """
  # Collect all unique server_ids
  server_ids = set()
  for task in tasks_list:
    for server in task.get('matched_servers', []):
      sid = server.get('server_id')
      if sid:
        server_ids.add(sid)

  print(f"Found {len(server_ids)} unique server IDs to load")

  server_index = {}
  missing = 0
  for server_id in sorted(server_ids):
    file_path = os.path.join(mcp_servers_dir, f"{server_id}.json")
    if not os.path.exists(file_path):
      missing += 1
      continue
    try:
      with open(file_path, 'r') as f:
        data = json.load(f)
      server_index[server_id] = data
    except (json.JSONDecodeError, OSError) as e:
      print(f"Warning: Could not load {file_path}: {e}")
      missing += 1

  print(f"Loaded {len(server_index)} server metadata files ({missing} missing/failed)")
  return server_index


def get_valid_onet_codes(occupation_to_tasks, num_tasks):
  """
  Filter occupations that have at least num_tasks tasks with matched servers.

  Args:
    occupation_to_tasks: Index from onet_soc_code to tasks
    num_tasks: Minimum number of tasks required

  Returns:
    list: List of valid onet_soc_codes
  """
  valid_onet_codes = [
    code for code, tasks in occupation_to_tasks.items()
    if len(tasks) >= num_tasks
  ]

  print(f"Found {len(valid_onet_codes)} occupations with >= {num_tasks} tasks with matched servers")
  return valid_onet_codes


def create_combos(occupation_to_tasks, valid_onet_codes, num_tasks, limit):
  """
  Create dataframe of all (occupation, task_combo) pairs.
  Sorted by occupation code, then by task_id within each occupation.

  Args:
    occupation_to_tasks: Index from onet_soc_code to tasks
    valid_onet_codes: List of valid onet_soc_codes
    num_tasks: Number of tasks per combination
    limit: Maximum number of combinations to generate

  Returns:
    pd.DataFrame with columns: onet_code, task_indices, tasks
  """
  combos = []
  for onet_code in sorted(valid_onet_codes):
    tasks = occupation_to_tasks[onet_code]
    # Sort tasks by task_id for determinism
    tasks_sorted = sorted(tasks, key=lambda t: t.get('task_id', ''))
    for combo in itertools.combinations(range(len(tasks_sorted)), num_tasks):
      combos.append({
        'onet_code': onet_code,
        'task_indices': combo,
        'tasks': [tasks_sorted[i] for i in combo]
      })
      if len(combos) >= limit:
        break
    if len(combos) >= limit:
      break

  df = pd.DataFrame(combos)
  print(f"Built {len(df)} (occupation, task_combo) combinations")
  return df


################
# Formatting Functions
################
def format_tasks_list(tasks):
  """
  Format O*NET tasks as a numbered list.

  Args:
    tasks: List of task records

  Returns:
    str: Formatted numbered list of tasks
  """
  lines = []
  for i, task in enumerate(tasks, 1):
    task_id = task.get('task_id', '?')
    task_text = task.get('task', '')
    lines.append(f"  {i}. [Task {task_id}] {task_text}")
  return '\n'.join(lines)


def format_knowledge(knowledge_items):
  """
  Format knowledge domains as a bulleted list.

  Args:
    knowledge_items: List of {'name': ..., 'importance': ...}

  Returns:
    str: Formatted bulleted list
  """
  if not knowledge_items:
    return "  - (No knowledge domain data available)"
  return '\n'.join(f"  - {item['name']} (importance: {item['importance']:.2f})" for item in knowledge_items)


def format_skills(skill_items):
  """
  Format skills as a bulleted list.

  Args:
    skill_items: List of {'name': ..., 'importance': ...}

  Returns:
    str: Formatted bulleted list
  """
  if not skill_items:
    return "  - (No skills data available)"
  return '\n'.join(f"  - {item['name']} (importance: {item['importance']:.2f})" for item in skill_items)


def format_tool_descriptions(tools_by_server, server_index):
  """
  Format tool descriptions from server metadata.

  Args:
    tools_by_server: dict mapping server_id to list of tool dicts from that server
    server_index: dict mapping server_id to full server metadata

  Returns:
    str: Formatted tool descriptions
  """
  tool_descs = []
  tool_num = 1

  for server_id, server_tools in tools_by_server.items():
    server_data = server_index.get(server_id, {})
    server_info = server_data.get('server', {})
    server_name = server_info.get('displayName', server_info.get('qualifiedName', 'Unknown Server'))
    server_desc = server_info.get('description', 'No description available')

    for tool in server_tools:
      tool_name = tool.get('name', 'Unknown Tool')
      tool_desc = tool.get('description', 'No description available')
      input_schema = tool.get('inputSchema', {})

      if input_schema:
        input_schema_str = f"```json\n{json.dumps(input_schema, indent=2)}\n```"
      else:
        input_schema_str = "(Not available)"

      desc = f"""### Tool {tool_num}: {tool_name}
**Server**: {server_name}
**Server Description**: {server_desc}
**Description**: {tool_desc}
**Input Schema**:
{input_schema_str}
"""
      tool_descs.append(desc)
      tool_num += 1

  return '\n'.join(tool_descs)


################
# Output Setup
################
def create_output_dir(args, output_foldername):
  """
  Create output directory and return the output file path and base directory.

  Args:
    args: Parsed arguments with output_folder and job_name
    output_foldername: Base name for the output folder/file

  Returns:
    tuple: (output_file_path, base_dir)
  """
  base = os.path.join(args.output_folder, args.job_name or output_foldername)
  os.makedirs(base, exist_ok=True)
  output_file_path = os.path.join(base, f"{output_foldername}_prepared.jsonl")
  return output_file_path, base


################
# Main Script
################
if __name__ == "__main__":
  args = get_args()

  print(f"Tool Use Question Generation (O*NET Task-first Approach)")
  print(f"Arguments:\n{args}")

  #################
  # Set random seed
  #################
  if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)

  #################
  # Load data
  #################
  tasks_to_servers_path = "tasks_to_smithery_servers.jsonl"
  occupation_data_path = "onet_db_30_1_text/Occupation Data.txt"
  knowledge_path = "onet_db_30_1_text/Knowledge.txt"
  skills_path = "onet_db_30_1_text/Skills.txt"
  prompt_template_path = "prompts/genq_from_onet_tasks.md"

  # Load tasks to servers mapping
  print(f"Loading tasks data from {tasks_to_servers_path}...")
  tasks_list = load_jsonl_to_list(tasks_to_servers_path)
  print(f"Loaded {len(tasks_list)} task records")

  # Load occupation data
  print(f"Loading occupation data from {occupation_data_path}...")
  occupation_dict = load_occupation_data(occupation_data_path)

  # Load knowledge and skills
  print(f"Loading knowledge domains from {knowledge_path}...")
  occupation_knowledge = load_onet_attribute(knowledge_path, args.top_k_knowledge)

  print(f"Loading skills from {skills_path}...")
  occupation_skills = load_onet_attribute(skills_path, args.top_k_skills)

  # Load prompt template
  print(f"Loading prompt template from {prompt_template_path}...")
  prompt_template = load_prompt_template(prompt_template_path)

  #################
  # Build indices
  #################
  occupation_to_tasks = create_occupation_to_tasks_index(tasks_list)
  valid_onet_codes = get_valid_onet_codes(occupation_to_tasks, args.num_tasks)

  if len(valid_onet_codes) == 0:
    raise ValueError(f"No occupations found with >= {args.num_tasks} tasks with matched servers. "
            f"Try reducing --num_tasks.")

  # Build server metadata index
  print(f"Loading MCP server metadata from {args.mcp_servers_dir}...")
  server_index = create_server_metadata_index(tasks_list, args.mcp_servers_dir)

  #################
  # Build all combinations (deterministic)
  #################
  task_combos = create_combos(occupation_to_tasks, valid_onet_codes, args.num_tasks, limit=args.total_prompts)

  if args.total_prompts > len(task_combos):
    print(f"Warning: Requested {args.total_prompts} prompts but only {len(task_combos)} combinations available.")
    print(f"Generating {len(task_combos)} prompts instead.")
    total_prompts = len(task_combos)
  else:
    total_prompts = args.total_prompts

  #################
  # Create output file / folder
  #################
  output_dirname = f"onet_tasks_{args.num_tasks}_tasks_{args.total_prompts}_{args.timestamp}"
  output_file_path, output_base_dir = create_output_dir(args, output_dirname)

  # Save arguments
  args_dict = vars(args)
  args_file_path = os.path.join(output_base_dir, "generation_args.json")
  with open(args_file_path, "w") as f:
    json.dump(args_dict, f, indent=2)
  print(f"Arguments saved to: {args_file_path}")

  #################
  # Save combinations dataframe as parquet
  #################
  combos = []
  for idx, row in task_combos.iterrows():
    combos.append({
      'combo_idx': idx,
      'onet_code': row['onet_code'],
      'task_indices': list(row['task_indices']),
      'tasks_metadata': [
        {
          'task_id': t.get('task_id'),
          'task': t.get('task'),
          'matched_servers': [s.get('server_id') for s in t.get('matched_servers', [])]
        }
        for t in row['tasks']
      ]
    })

  combos_df = pd.DataFrame(combos)
  combos_parquet_filepath = os.path.join(output_base_dir, "combos.parquet")
  combos_df.to_parquet(combos_parquet_filepath, index=False)
  print(f"Combinations dataframe saved to: {combos_parquet_filepath}")

  #################
  # Generate outputs (deterministic - iterate over first N combinations)
  #################
  results = []

  pbar = tqdm(total=total_prompts, desc="Generating prompts")
  for i in range(total_prompts):
    row = task_combos.iloc[i]
    onet_code = row['onet_code']
    task_combo = row['tasks']  # List of task records

    occupation_info = occupation_dict.get(onet_code, {'title': 'Unknown Occupation', 'description': ''})
    occupation_title = occupation_info['title']
    occupation_description = occupation_info['description']

    # Collect all unique server_ids across this combo's tasks' matched_servers
    # and map server_id -> list of tools from that server
    seen_server_ids = set()
    for task in task_combo:
      for server in task.get('matched_servers', []):
        seen_server_ids.add(server.get('server_id'))

    # Build tools_by_server: server_id -> list of tool dicts from server metadata
    tools_by_server = {}
    mcp_servers_metadata = []
    for server_id in sorted(seen_server_ids):
      server_data = server_index.get(server_id)
      if server_data is None:
        continue
      server_info = server_data.get('server', {})
      server_tools = server_info.get('tools', [])
      if server_tools:
        tools_by_server[server_id] = server_tools

      # Build metadata for output (compatible with tool-first format)
      mcp_servers_metadata.append({
        "server_id": server_id,
        "server_name": server_info.get('displayName', server_info.get('qualifiedName', '')),
        "server_description": server_info.get('description', ''),
        "tools": [
          {
            "name": t.get('name'),
            "description": t.get('description'),
            "inputSchema": t.get('inputSchema')
          }
          for t in server_tools
        ],
        "source_file_path": os.path.join(args.mcp_servers_dir, f"{server_id}.json")
      })

    # Look up knowledge and skills for this occupation
    knowledge_items = occupation_knowledge.get(onet_code, [])
    skills_items = occupation_skills.get(onet_code, [])

    # Format template placeholders
    tasks_str = format_tasks_list(task_combo)
    knowledge_str = format_knowledge(knowledge_items)
    skills_str = format_skills(skills_items)
    tool_descriptions_str = format_tool_descriptions(tools_by_server, server_index)

    # Fill in the template
    prompt = prompt_template.replace("{NUM_TASKS}", str(args.num_tasks))
    prompt = prompt.replace("{OCCUPATION}", occupation_title)
    prompt = prompt.replace("{OCCUPATION_DESCRIPTION}", occupation_description)
    prompt = prompt.replace("{KNOWLEDGE}", knowledge_str)
    prompt = prompt.replace("{SKILLS}", skills_str)
    prompt = prompt.replace("{TASKS}", tasks_str)
    prompt = prompt.replace("{TOOL_DESCRIPTIONS}", tool_descriptions_str)

    # Build matched_servers list for metadata
    matched_servers_meta = []
    for server_id in sorted(seen_server_ids):
      server_data = server_index.get(server_id)
      if server_data is None:
        continue
      server_info = server_data.get('server', {})
      matched_servers_meta.append({
        "server_id": server_id,
        "server_name": server_info.get('displayName', server_info.get('qualifiedName', ''))
      })

    # Create result
    result = {
      "messages": [
        {
          "role": "user",
          "content": prompt
        }
      ],
      "metadata": {
        "prompt_id": f"{i:08d}",
        "row_id": i,
        "mode": "onet_tasks",
        "question_gen_args": args_dict,
        "onet_soc_code": onet_code,
        "occupation_title": occupation_title,
        "tasks": [
          {
            "task_id": t.get('task_id'),
            "task": t.get('task')
          }
          for t in task_combo
        ],
        "matched_servers": matched_servers_meta,
        "mcp_servers": mcp_servers_metadata
      }
    }

    results.append(result)
    pbar.update(1)

  pbar.close()

  # Save results
  with open(output_file_path, "w") as f:
    for result in results:
      f.write(json.dumps(result) + "\n")

  print(f"Finished. Total prompts: {len(results)}")
  print(f"Total combinations available: {len(task_combos)}")
  print(f"Output file: {output_file_path}")
  print(f"Combinations parquet: {combos_parquet_filepath}")
