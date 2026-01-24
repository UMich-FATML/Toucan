import os
import sys
import argparse
import json
import time
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils import load_jsonl_to_list

################
# Use Cases
################
"""
This script generates questions for tool use scenarios using occupation-based tool sampling via O*NET codes.

Example Usage:

1. Basic usage - generate 100 prompts with 2 tools each:
   python step1.1_gen_questions_onet.py --num_tools 2 --total_prompts 100

2. Generate with specific seed for reproducibility:
   python step1.1_gen_questions_onet.py --num_tools 3 --total_prompts 500 --seed 42

3. Custom output folder and job name:
   python step1.1_gen_questions_onet.py --num_tools 2 --total_prompts 1000 --output_folder ../data --job_name my_experiment

Key Parameters:
- --num_tools: Number of tools to include in each prompt (required)
- --total_prompts: Total number of prompts to generate (required)
- --seed: Random seed for reproducibility
"""

################
# Configurations
################
def get_args():
  parser = argparse.ArgumentParser(description="Tool Use Question Generation using O*NET Occupation-based Sampling.")

  # Required parameters
  parser.add_argument("--num_tools", type=int, required=True, help="Number of tools per prompt.")
  parser.add_argument("--total_prompts", type=int, required=True, help="Total number of prompts to generate.")

  # Optional parameters
  parser.add_argument("--output_folder", type=str, default="../data", help="Output folder path.")
  parser.add_argument("--job_name", type=str, default=None, help="Job name for organization.")
  parser.add_argument("--timestamp", type=int, default=int(time.time()), help="Timestamp for the job.")
  parser.add_argument("--seed", type=int, default=None, help="Random seed.")

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
  occupation_lookup = {}

  with open(occupation_file_path, 'r', encoding='utf-8') as f:
    # Skip header
    header = f.readline()

    for line in f:
      parts = line.strip().split('\t')
      if len(parts) >= 3:
        code = parts[0]
        title = parts[1]
        description = parts[2]
        occupation_lookup[code] = {
          'title': title,
          'description': description
        }

  print(f"Loaded {len(occupation_lookup)} occupations from {occupation_file_path}")
  return occupation_lookup


def load_prompt_template(template_path):
  """Load the prompt template from file."""
  with open(template_path, 'r', encoding='utf-8') as f:
    return f.read()


################
# Index Building Functions
################
def build_occupation_to_tools_index(tools_data):
  """
  Build an index from O*NET SOC codes to tool keys (server_idx, tool_idx).

  A tool belongs to an occupation if any of its matched_tasks has that code.

  Args:
    tools_data: List of tool records from tools_to_tasks.jsonl

  Returns:
    dict: Mapping from onet_soc_code to list of (server_idx, tool_idx, tool_record)
  """
  occupation_to_tools = defaultdict(list)

  for tool in tools_data:
    server_idx = tool['server_idx']
    tool_idx = tool['tool_idx']
    matched_tasks = tool.get('matched_tasks', [])

    # Track which occupations this tool belongs to (avoid duplicates)
    seen_codes = set()

    for task in matched_tasks:
      onet_code = task.get('onet_soc_code')
      if onet_code and onet_code not in seen_codes:
        seen_codes.add(onet_code)
        occupation_to_tools[onet_code].append({
          'server_idx': server_idx,
          'tool_idx': tool_idx,
          'tool_record': tool
        })

  print(f"Built index with {len(occupation_to_tools)} occupations")
  return occupation_to_tools


def get_valid_occupations(occupation_to_tools, num_tools):
  """
  Filter occupations that have at least num_tools tools.

  Args:
    occupation_to_tools: Index from onet_soc_code to tools
    num_tools: Minimum number of tools required

  Returns:
    list: List of valid onet_soc_codes
  """
  valid_occupations = [
    code for code, tools in occupation_to_tools.items()
    if len(tools) >= num_tools
  ]

  print(f"Found {len(valid_occupations)} occupations with >= {num_tools} tools")
  return valid_occupations


################
# Sampling Functions
################
def sample_tools_for_occupation(occupation_to_tools, onet_code, num_tools):
  """
  Sample tools from a specific occupation.

  Args:
    occupation_to_tools: Index from onet_soc_code to tools
    onet_code: The O*NET SOC code to sample from
    num_tools: Number of tools to sample

  Returns:
    list: List of sampled tool records with their tasks for this occupation
  """
  available_tools = occupation_to_tools[onet_code]
  sampled = random.sample(available_tools, num_tools)

  # For each sampled tool, collect tasks that match this occupation
  result = []
  for tool_entry in sampled:
    tool_record = tool_entry['tool_record']

    # Filter tasks that belong to this occupation
    relevant_tasks = [
      task for task in tool_record.get('matched_tasks', [])
      if task.get('onet_soc_code') == onet_code
    ]

    result.append({
      'server_idx': tool_entry['server_idx'],
      'tool_idx': tool_entry['tool_idx'],
      'tool_record': tool_record,
      'relevant_tasks': relevant_tasks
    })

  return result


################
# Formatting Functions
################
def format_tasks(sampled_tools):
  """
  Format tasks as a bulleted list.

  Args:
    sampled_tools: List of sampled tool entries with relevant_tasks

  Returns:
    str: Formatted bulleted list of tasks
  """
  tasks_set = set()  # Avoid duplicate tasks

  for tool_entry in sampled_tools:
    for task in tool_entry.get('relevant_tasks', []):
      task_text = task.get('task', '')
      if task_text:
        tasks_set.add(task_text)

  if not tasks_set:
    return "  - (No specific tasks available)"

  # Format as bulleted list
  return '\n'.join(f"  - {task}" for task in sorted(tasks_set))


def format_tool_descriptions(sampled_tools, onet_code):
  """
  Format tool descriptions with name, description, relevant tasks, and server info.

  Args:
    sampled_tools: List of sampled tool entries
    onet_code: The O*NET SOC code for filtering relevant tasks

  Returns:
    str: Formatted tool descriptions
  """
  descriptions = []

  for i, tool_entry in enumerate(sampled_tools, 1):
    tool_record = tool_entry['tool_record']
    relevant_tasks = tool_entry.get('relevant_tasks', [])

    tool_name = tool_record.get('tool_name', 'Unknown Tool')
    tool_desc = tool_record.get('tool_description', 'No description available')
    server_name = tool_record.get('server_name', 'Unknown Server')
    server_analysis = tool_record.get('server_analysis', 'No analysis available')

    # Format relevant tasks
    if relevant_tasks:
      tasks_formatted = '\n'.join(f"  - {task.get('task', '')}" for task in relevant_tasks if task.get('task'))
    else:
      tasks_formatted = "  - (No specific tasks for this occupation)"

    tool_section = f"""### Tool {i}: {tool_name}
**Server**: {server_name}
**Server Analysis**: {server_analysis}
**Description**: {tool_desc}
**Relevant Tasks for this Occupation**:
{tasks_formatted}
"""
    descriptions.append(tool_section)

  return '\n'.join(descriptions)


################
# MCP Server Metadata Functions
################
def build_mcp_servers_metadata(sampled_tools, mcp_servers_dir):
  """
  Load MCP server metadata from JSON files.

  Args:
    sampled_tools: List of sampled tool entries
    mcp_servers_dir: Path to MCP servers directory

  Returns:
    list: List of MCP server metadata dictionaries
  """
  mcp_servers = []
  seen_servers = set()  # Track unique servers

  for tool_entry in sampled_tools:
    tool_record = tool_entry['tool_record']
    server_filename = tool_record.get('server_filename', '')

    if not server_filename or server_filename in seen_servers:
      continue

    seen_servers.add(server_filename)

    file_path = os.path.join(mcp_servers_dir, server_filename)

    try:
      with open(file_path, 'r') as f:
        mcp_data = json.load(f)

      server_metadata = mcp_data.get('metadata', {})
      server_info = server_metadata.get('server_info_crawled', {})
      remote_response = server_metadata.get('remote_server_response', {})

      mcp_servers.append({
        # Core identifiers
        "server_id": server_metadata.get('server_id'),
        "server_name": server_metadata.get('server_name'),
        "rank_by_usage": server_metadata.get('rank_by_usage'),

        "server_info": server_info,
        "remote_server_response": remote_response,
        "labels": mcp_data.get('labels', {}),

        # File paths and processing info
        "original_file": server_metadata.get('original_file', ''),
        "source_file_path": file_path,
        "source_filename": server_filename,
        "processed_timestamp": server_metadata.get('processed_timestamp'),
        "processing_source": server_metadata.get('processing_source', ''),
        "rank": server_metadata.get('rank')
      })

    except (json.JSONDecodeError, FileNotFoundError) as e:
      print(f"Warning: Could not load MCP server file {file_path}: {e}")
      continue

  return mcp_servers


################
# Main Script
################
if __name__ == "__main__":
  args = get_args()

  print(f"Tool Use Question Generation (O*NET Occupation-based Sampling)")
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
  # Paths relative to datagen directory
  tools_to_tasks_path = "tools_to_tasks.jsonl"
  occupation_data_path = "onet_db_30_1_text/Occupation Data.txt"
  prompt_template_path = "prompts/genq_from_tools_onet.md"
  mcp_servers_dir = "../mcp_servers"

  # Load tools to tasks mapping
  print(f"Loading tools data from {tools_to_tasks_path}...")
  tools_data = load_jsonl_to_list(tools_to_tasks_path)
  print(f"Loaded {len(tools_data)} tool records")

  # Load occupation data
  print(f"Loading occupation data from {occupation_data_path}...")
  occupation_lookup = load_occupation_data(occupation_data_path)

  # Load prompt template
  print(f"Loading prompt template from {prompt_template_path}...")
  prompt_template = load_prompt_template(prompt_template_path)

  #################
  # Build index and filter
  #################
  occupation_to_tools = build_occupation_to_tools_index(tools_data)
  valid_occupations = get_valid_occupations(occupation_to_tools, args.num_tools)

  if len(valid_occupations) == 0:
    raise ValueError(f"No occupations found with >= {args.num_tools} tools. "
            f"Try reducing --num_tools.")

  #################
  # Create output file / folder
  #################
  output_filename = f"ToolUse_s2q_onet_{args.total_prompts}_{args.num_tools}tool_{args.timestamp}_prepared.jsonl"
  output_foldername = f"ToolUse_onet_{args.total_prompts}_{args.num_tools}tool_{args.timestamp}"

  if not args.job_name:
    if not os.path.exists(args.output_folder):
      os.makedirs(args.output_folder)
    if not os.path.exists(f"{args.output_folder}/{output_foldername}"):
      os.makedirs(f"{args.output_folder}/{output_foldername}")
    output_dir = f"{args.output_folder}/{output_foldername}/{output_filename}"
    args_output_dir = f"{args.output_folder}/{output_foldername}"
  else:
    if not os.path.exists(f"{args.output_folder}/{args.job_name}"):
      os.makedirs(f"{args.output_folder}/{args.job_name}")
    output_dir = f"{args.output_folder}/{args.job_name}/{output_filename}"
    args_output_dir = f"{args.output_folder}/{args.job_name}"

  # Save arguments
  args_dict = vars(args)
  args_file_path = f"{args_output_dir}/generation_args.json"
  with open(args_file_path, "w") as f:
    json.dump(args_dict, f, indent=2)
  print(f"Arguments saved to: {args_file_path}")

  #################
  # Generate outputs
  #################
  results = []

  pbar = tqdm(total=args.total_prompts, desc="Generating prompts")
  for i in range(args.total_prompts):
    # Randomly select a valid occupation
    onet_code = random.choice(valid_occupations)
    occupation_info = occupation_lookup.get(onet_code, {'title': 'Unknown Occupation', 'description': ''})
    occupation_title = occupation_info['title']

    # Sample tools from this occupation
    sampled_tools = sample_tools_for_occupation(occupation_to_tools, onet_code, args.num_tools)

    # Format template placeholders
    tasks_formatted = format_tasks(sampled_tools)
    tool_descriptions_formatted = format_tool_descriptions(sampled_tools, onet_code)

    # Fill in the template
    seed_prompt = prompt_template.replace("{NUM_TOOLS}", str(args.num_tools))
    seed_prompt = seed_prompt.replace("{OCCUPATION}", occupation_title)
    seed_prompt = seed_prompt.replace("{TASKS}", tasks_formatted)
    seed_prompt = seed_prompt.replace("{TOOL_DESCRIPTIONS}", tool_descriptions_formatted)

    # Build MCP server metadata
    mcp_servers_metadata = build_mcp_servers_metadata(sampled_tools, mcp_servers_dir)

    # Create result in the same format as step1.1_gen_questions.py
    result = {
      "messages": [
        {
          "role": "user",
          "content": seed_prompt
        }
      ],
      "metadata": {
        "prompt_id": f"{i:08d}",
        "row_id": i,
        "mode": "onet_occupation",
        "question_gen_args": args_dict,
        "onet_soc_code": onet_code,
        "occupation_title": occupation_title,
        "sampled_tools": [
          {
            "server_idx": t['server_idx'],
            "tool_idx": t['tool_idx'],
            "tool_name": t['tool_record'].get('tool_name'),
            "server_name": t['tool_record'].get('server_name')
          }
          for t in sampled_tools
        ],
        "mcp_servers": mcp_servers_metadata
      }
    }

    results.append(result)
    pbar.update(1)

  pbar.close()

  # Save results
  with open(output_dir, "w") as f:
    for result in results:
      f.write(json.dumps(result) + "\n")

  print(f"Finished. Total prompts: {len(results)}")
  print(f"Output file: {output_dir}")
