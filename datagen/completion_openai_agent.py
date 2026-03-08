import torch
import os
import sys
import argparse
import copy
import json
import re
import types
import asyncio
import base64
import signal
import atexit
import shutil
from glob import glob
from time import time
from tqdm import tqdm
from virtual_tools import VirtualToolBackend, create_dynamic_virtual_tool

from utils import load_dataset_from_file, save_dataset, validate_api_pool_from_file, get_model_abbreviation


# Suppress OpenAI Agent SDK tracing logs before importing
import logging
logging.getLogger("openai.agents").setLevel(logging.ERROR)
logging.getLogger("agents").setLevel(logging.ERROR)
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

# OpenAI Agent imports
from agents.mcp import MCPServerStreamableHttp
from agents.run_context import RunContextWrapper
from agents import Agent, OpenAIResponsesModel, Runner, SQLiteSession
from agents.tracing import set_tracing_disabled
set_tracing_disabled(True)
from openai import AsyncClient, AsyncOpenAI
from typing import Dict, Any, List, Optional
from pydantic import create_model, Field, BaseModel

# Check if agents library is installed
try:
    import agents
except ImportError:
    print("agents library is not installed. Please install it.")
    exit(1)

# Global cleanup function for MCP resources
def cleanup_mcp_resources():
    """Clean up MCP resources on exit"""
    # Only cleanup if we're using agent mode
    try:
        # Check if args is available and agent mode is enabled
        if 'args' in globals() and hasattr(args, 'agent') and args.agent:
            # OpenAI Agent framework handles cleanup automatically
            pass
    except Exception as e:
        # print(f"⚠️ Warning: Emergency MCP cleanup failed: {e}")
        pass

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    # print(f"\n🛑 Received signal {signum}. Cleaning up...")
    cleanup_mcp_resources()
    # print("👋 Exiting gracefully.")
    os._exit(0)  # Use os._exit instead of sys.exit to avoid atexit conflicts

# Register cleanup functions
atexit.register(cleanup_mcp_resources)
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Response Generation Manager.")
    parser.add_argument("--model_path", type=str, default="openai/gpt-oss-120b",
                        help="Model path for inference")
    parser.add_argument("--input_file", type=str, default=None, help="Input dataset file name")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index (inclusive) of rows to process.")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional number of rows to process from start_idx.")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="OpenAI-compatible API base URL ending with /v1.")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key for the endpoint.")
    parser.add_argument("--smithery_api_key", type=str, default="", help="Smithery API Key")
    parser.add_argument("--smithery_profile", type=str, default="", help="Smithery Profile")
    parser.add_argument("--smithery_api_pool", type=str, default="smithery_api_pool.json", help="Path to Smithery API pool JSON file")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of parallel workers (default: use API pool size)")

    # Generation Parameters
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--step", type=str, default="unknown", help="Processing step identifier.")
    parser.add_argument("--agent", type=str, default="openai_agent", help="Use agent inference for items with MCP server URLs")
    parser.add_argument("--timeout", type=int, default=90, help="Timeout in seconds for each item processing (default: 90 seconds)")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for each item processing (default: 3)")
    parser.add_argument("--fncall_prompt_type", type=str, default="nous", help="Function call prompt type (default: nous)")
    parser.add_argument("--parallel_function_calls", type=bool, default=True, help="Parallel function calls (default: True)")
    parser.add_argument("--reasoning_effort", type=str, default="high", help="Reasoning effort (default: high)")
    parser.add_argument("--enable_tool_hint", action="store_true", help="Enable tool hint (default: off)")
    parser.add_argument("--enable_irrelevant_warning", action="store_true", help="Enable irrelevant warning (default: off)")
    parser.add_argument("--max_turns", type=int, default=10, help="Maximum number of turns for agent inference (default: 10)")

    #tool parameters
    parser.add_argument("--virtual_tools", action="store_true", help="Use LLM-hallucinated tools instead of real MCP connections")
    parser.add_argument("--mcp_server_dir", type=str, default="../mcp_servers/smithery_mcp_servers_0210",
                    help="Path to directory of MCP server JSON files (named {server_id}.json). "
                         "Enriches virtual tools with server analysis/descriptions from smithery files.")
    return parser.parse_args()

args = get_args()
print(f"Response Generation Manager. Arguments: {args}") # For logging

if args.input_file is None:
    raise ValueError("Please specify the input file path.")
    
# Input check: check if ends with prepared.jsonl or prepared.json
if not args.input_file.endswith("prepared.jsonl") and not args.input_file.endswith("prepared.json"):
    print("Error: Input file must end with prepared.json(l) for completion pipeline. Please make sure you are using the correct input file.")
    exit(1)
if args.start_idx < 0:
    raise ValueError("--start_idx must be a non-negative integer.")
if args.batch_size is not None and args.batch_size <= 0:
    raise ValueError("--batch_size must be a positive integer.")

normalized_base_url = args.base_url.rstrip("/").removesuffix("/chat/completions")
if not normalized_base_url.endswith("/v1"):
    raise ValueError("--base_url must end with /v1.")
args.base_url = normalized_base_url

INPUT_FILE_NAME = args.input_file 

model_abbreviation = get_model_abbreviation(args.model_path)
config_str = f"{model_abbreviation}_{args.reasoning_effort}_pfc" if args.parallel_function_calls else f"{model_abbreviation}_{args.reasoning_effort}_sfc"

# Global API pool variable
smithery_api_pool = None

# Row-id keyed map of full question-generation conversation histories.
# Populated once from parent-dir *_results.jsonl, matched via *_3sanitized.jsonl row_ids.
QUESTION_GEN_HISTORY_BY_ROW_ID = {}
QUESTION_GEN_HISTORY_SOURCE_FILE = None
QUESTION_GEN_HISTORY_READY = False

def load_and_validate_smithery_api_pool(pool_file_path):
    """
    Load Smithery API pool from JSON file.
    Non-blocking: If validation fails or file is missing, returns empty list/None
    instead of raising errors, allowing the script to proceed (e.g. for virtual tools).
    """
    global smithery_api_pool
    
    print("=" * 50)
    print("🔍 SMITHERY API POOL CHECK (Non-blocking)")
    print("=" * 50)
    
    try:
        # 1. Check if pool file exists
        if not os.path.exists(pool_file_path):
            print(f"ℹ️  API pool file {pool_file_path} not found.")
            print("   Proceeding without API pool (using args or virtual tools).")
            smithery_api_pool = []
            return []

        # 2. Try to validate (but don't crash if network fails)
        print(f"📁 Found {pool_file_path}. Attempting validation...")
        try:
            results = validate_api_pool_from_file(pool_file_path)
            
            if "error" in results:
                print(f"⚠️  API pool validation warning: {results['error']}")
                print("   Proceeding without verified pool.")
                smithery_api_pool = []
                return []
            
            # Load original data to get valid entries with API keys
            with open(pool_file_path, 'r') as f:
                original_data = json.load(f)
                original_pool = original_data.get('api_pool', [])
            
            # Keep only valid entries
            valid_pool = []
            for result in results['results']:
                if result['valid']:
                    for original_entry in original_pool:
                        if original_entry['profile'] == result['profile']:
                            valid_pool.append(original_entry)
                            break
            
            smithery_api_pool = valid_pool
            print(f"✅ Loaded {len(smithery_api_pool)} valid API keys from pool.")
            return smithery_api_pool

        except Exception as e:
            print(f"⚠️  Network/Validation check failed: {e}")
            print("   Proceeding without verified pool (this is fine for virtual tools).")
            smithery_api_pool = []
            return []

    except Exception as e:
        print(f"⚠️  Unexpected error loading pool: {e}")
        smithery_api_pool = []
        return []

def get_api_key_for_worker(worker_id):
    """Get API key and profile for a specific worker"""
    if smithery_api_pool and len(smithery_api_pool) > 0:
        # Round-robin assignment
        pool_entry = smithery_api_pool[worker_id % len(smithery_api_pool)]
        return pool_entry['api_key'], pool_entry['profile']
    else:
        return args.smithery_api_key, args.smithery_profile

def construct_mcp_server_url(server_info, api_key=None, profile=None):
    """
    Construct MCP server URL from server info.
    """
    if not server_info:
        return None
        
    server_url = server_info.get('python_sdk_url', '')
    if not server_url:
        return None
    
    # Use provided api_key and profile, or fall back to args
    if api_key is None:
        api_key = args.smithery_api_key
    if profile is None:
        profile = args.smithery_profile
    
    # Get or create default config
    mcp_config = server_info.get('python_sdk_config', "")
    if mcp_config == "":
        mcp_config = {"debug": False}
    else:
        try:
            mcp_config = json.loads(mcp_config)
        except json.JSONDecodeError:
            mcp_config = {"debug": False}
    
    # Replace URL placeholders
    config_b64 = base64.b64encode(json.dumps(mcp_config).encode()).decode()
    if "{config_b64}" in server_url:
        server_url = server_url.replace("{config_b64}", config_b64)
    if "{smithery_api_key}" in server_url:
        server_url = server_url.replace("{smithery_api_key}", api_key)
    if "{smithery_profile}" in server_url:
        server_url = server_url.replace("{smithery_profile}", profile)
    elif "&profile=" not in server_url and "profile=" not in server_url:
        server_url += f"&profile={profile}"
    
    return server_url


def normalize_row_id_key(row_id):
    """Normalize row_id to a stable string key for joins."""
    if row_id is None:
        return None
    try:
        return str(int(row_id))
    except (ValueError, TypeError):
        return str(row_id)


def resolve_sanitized_file(prepared_input_file):
    """
    Resolve the sibling *_3sanitized.jsonl corresponding to a prepared input file.
    Falls back to a deterministic match in the same output directory.
    """
    output_dir = os.path.dirname(prepared_input_file) or "."
    input_basename = os.path.basename(prepared_input_file)
    stem, _ = os.path.splitext(input_basename)

    if stem.endswith("_4prepared"):
        base_stem = stem[:-10]
    elif stem.endswith("_prepared"):
        base_stem = stem[:-9]
    else:
        base_stem = stem

    exact_candidate = os.path.join(output_dir, f"{base_stem}_3sanitized.jsonl")
    if os.path.exists(exact_candidate):
        return exact_candidate

    prefixed = sorted(glob(os.path.join(output_dir, f"{base_stem}*_3sanitized.jsonl")))
    if prefixed:
        return prefixed[0]

    any_sanitized = sorted(glob(os.path.join(output_dir, "*_3sanitized.jsonl")))
    if any_sanitized:
        return any_sanitized[0]

    return None


def load_sanitized_row_id_keys(sanitized_file):
    """Load row_id keys from *_3sanitized.jsonl."""
    row_id_keys = set()
    if not sanitized_file or not os.path.exists(sanitized_file):
        return row_id_keys

    with open(sanitized_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = normalize_row_id_key((item.get("metadata") or {}).get("row_id"))
            if key is not None:
                row_id_keys.add(key)
    return row_id_keys


def count_results_row_id_overlap(results_file, row_id_keys):
    """Count row_id overlap between a *_results.jsonl file and sanitized row ids."""
    if not row_id_keys:
        return 0

    overlap = 0
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = normalize_row_id_key((item.get("metadata") or {}).get("row_id"))
            if key is not None and key in row_id_keys:
                overlap += 1
    return overlap


def select_results_file(parent_dir, sanitized_file, row_id_keys):
    """
    Select parent-dir *_results.jsonl that best matches sanitized row_ids.
    """
    candidates = sorted(glob(os.path.join(parent_dir, "*_results.jsonl")))
    if not candidates:
        return None, 0

    expected_file = None
    if sanitized_file:
        sanitized_name = os.path.basename(sanitized_file)
        if sanitized_name.endswith("_3sanitized.jsonl"):
            prefix = sanitized_name[:-len("_3sanitized.jsonl")]
            expected_path = os.path.join(parent_dir, f"{prefix}_results.jsonl")
            if os.path.exists(expected_path):
                expected_file = expected_path

    scored = []
    for path in candidates:
        overlap = count_results_row_id_overlap(path, row_id_keys)
        expected_bonus = 1 if expected_file and path == expected_file else 0
        scored.append((overlap, expected_bonus, path))

    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    best_overlap, _, best_file = scored[0]
    return best_file, best_overlap


def load_question_generation_history_by_row_id(prepared_input_file):
    """
    Load row_id -> full messages map using:
      - row ids from sibling *_3sanitized.jsonl
      - full histories from parent-dir *_results.jsonl
    """
    output_dir = os.path.dirname(prepared_input_file) or "."
    parent_dir = os.path.dirname(output_dir) or "."

    sanitized_file = resolve_sanitized_file(prepared_input_file)
    if not sanitized_file:
        print("⚠️ Could not find *_3sanitized.jsonl near input; question-gen history lookup disabled.")
        return {}, None

    row_id_keys = load_sanitized_row_id_keys(sanitized_file)
    if not row_id_keys:
        print(f"⚠️ No row_id values found in sanitized file: {sanitized_file}")
        return {}, None

    results_file, overlap = select_results_file(parent_dir, sanitized_file, row_id_keys)
    if not results_file:
        print(f"⚠️ No parent-dir *_results.jsonl found in: {parent_dir}")
        return {}, None

    history_by_row_id = {}
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = normalize_row_id_key((item.get("metadata") or {}).get("row_id"))
            if key is None or key not in row_id_keys:
                continue

            messages = item.get("messages")
            if isinstance(messages, list):
                history_by_row_id[key] = messages

    print("🧭 Question-generation history lookup:")
    print(f"   - Sanitized file: {sanitized_file}")
    print(f"   - Parent results file: {results_file}")
    print(f"   - Sanitized row_ids: {len(row_id_keys)}")
    print(f"   - Row-id overlap: {overlap}")
    print(f"   - Loaded histories: {len(history_by_row_id)}")
    return history_by_row_id, results_file


def ensure_question_generation_history_loaded():
    """Initialize row_id->messages history cache once per run."""
    global QUESTION_GEN_HISTORY_BY_ROW_ID, QUESTION_GEN_HISTORY_SOURCE_FILE, QUESTION_GEN_HISTORY_READY
    if QUESTION_GEN_HISTORY_READY:
        return

    QUESTION_GEN_HISTORY_BY_ROW_ID, QUESTION_GEN_HISTORY_SOURCE_FILE = (
        load_question_generation_history_by_row_id(args.input_file)
    )
    QUESTION_GEN_HISTORY_READY = True

def convert_openai_agent_result_to_messages(result, original_messages, system_prompt=None):
    """Convert OpenAI Agent result to message format compatible with Qwen Agent structure"""
    all_messages = []

    # Prepend system prompt if provided
    if system_prompt:
        all_messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # Add original user message
    all_messages.extend(original_messages)
    
    # Process conversation flow from OpenAI Agent
    if hasattr(result, 'new_items') and result.new_items:
        current_reasoning = []  # Collect reasoning content
        matched_call_ids = set()  # Track which tool calls have been matched to outputs

        for item_flow in result.new_items:
            if item_flow.type == "reasoning_item":
                # Collect reasoning content
                if hasattr(item_flow, 'raw_item') and hasattr(item_flow.raw_item, 'content'):
                    for content in item_flow.raw_item.content:
                        if hasattr(content, 'text'):
                            current_reasoning.append(content.text)

            elif item_flow.type == "tool_call_item":
                # Extract tool call information
                if hasattr(item_flow, 'raw_item'):
                    tool_call = {
                        "name": getattr(item_flow.raw_item, 'name', None),
                        "arguments": getattr(item_flow.raw_item, 'arguments', None),
                        "call_id": getattr(item_flow.raw_item, 'call_id', None)
                    }

                    # Flush reasoning as a separate assistant message before the tool call
                    if current_reasoning:
                        all_messages.append({
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "\n".join(current_reasoning)
                        })
                        current_reasoning = []  # Reset for next iteration

                    # Create assistant message with tool call only
                    assistant_msg = {
                        "role": "assistant",
                        "content": "",
                        "function_call": tool_call
                    }

                    all_messages.append(assistant_msg)

            elif item_flow.type == "tool_call_output_item":
                # Extract tool output
                if hasattr(item_flow, 'output'):
                    try:
                        # Parse the JSON output
                        output_data = json.loads(item_flow.output)
                        if output_data.get('type') == 'text':
                            # Parse the inner JSON text
                            inner_data = json.loads(output_data.get('text', '{}'))
                            tool_output = json.dumps(inner_data)
                        else:
                            tool_output = item_flow.output
                    except:
                        tool_output = item_flow.output

                    # Find the corresponding tool call name from previous messages
                    tool_name = 'unknown'
                    matched_call_id = None
                    if hasattr(item_flow, 'raw_item'):
                        raw = item_flow.raw_item
                        call_id = None
                        for attr in ['tool_call_id', 'call_id', 'id', 'toolCallId']:
                            if hasattr(raw, attr):
                                call_id = getattr(raw, attr)
                                break
                        if call_id is not None:
                            # Look for the matching tool call in previous messages
                            for prev_msg in reversed(all_messages):
                                if (prev_msg.get('role') == 'assistant' and
                                    'function_call' in prev_msg and
                                    prev_msg['function_call'].get('call_id') == call_id):
                                    tool_name = prev_msg['function_call'].get('name', 'unknown')
                                    matched_call_id = call_id
                                    break

                    # Fallback: find the oldest unmatched assistant function_call
                    if tool_name == 'unknown':
                        for prev_msg in all_messages:
                            if prev_msg.get('role') == 'assistant' and 'function_call' in prev_msg:
                                fc = prev_msg['function_call']
                                fc_call_id = fc.get('call_id')
                                if fc_call_id not in matched_call_ids:
                                    name_candidate = fc.get('name')
                                    if name_candidate:
                                        tool_name = name_candidate
                                        matched_call_id = fc_call_id
                                        break

                    if matched_call_id is not None:
                        matched_call_ids.add(matched_call_id)

                    all_messages.append({
                        "role": "function",
                        "content": tool_output,
                        "name": tool_name
                    })
            
            elif item_flow.type == "message_output_item":
                # Extract final assistant message
                if hasattr(item_flow, 'raw_item') and hasattr(item_flow.raw_item, 'content'):
                    message_texts = []
                    for content in item_flow.raw_item.content:
                        if hasattr(content, 'text'):
                            message_texts.append(content.text)
                    
                    # Flush any remaining reasoning content as a separate message
                    if current_reasoning:
                        all_messages.append({
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "\n".join(current_reasoning)
                        })
                        current_reasoning = []
                    
                    # Create final assistant message
                    final_content = "\n".join(message_texts)
                    if final_content.strip():
                        final_msg = {
                            "role": "assistant",
                            "content": final_content
                        }
                        all_messages.append(final_msg)
    
    # If no conversation flow items, use final output  
    new_messages_start = len(original_messages) + (1 if system_prompt else 0)
    if not any(msg.get('role') == 'assistant' and msg.get('content') for msg in all_messages[new_messages_start:]):
        final_msg = {
            "role": "assistant", 
            "content": result.final_output
        }
        
        # Try to extract reasoning from the result if available
        reasoning_content = []
        if hasattr(result, 'new_items') and result.new_items:
            for item_flow in result.new_items:
                if item_flow.type == "reasoning_item":
                    if hasattr(item_flow, 'raw_item') and hasattr(item_flow.raw_item, 'content'):
                        for content in item_flow.raw_item.content:
                            if hasattr(content, 'text'):
                                reasoning_content.append(content.text)
        
        # Emit reasoning as a separate assistant message before the final message
        if reasoning_content:
            all_messages.append({
                "role": "assistant",
                "content": "",
                "reasoning_content": "\n".join(reasoning_content)
            })
        
        all_messages.append(final_msg)
    
    return all_messages

def construct_mcp_url_from_source(server_info, api_key=None, profile=None):
    """
    Construct MCP server URL from step 1.1 format data.
    This handles the case where server_info has source_file_path or server_id
    instead of a nested server_info dict with python_sdk_url.
    """
    if api_key is None:
        api_key = args.smithery_api_key
    if profile is None:
        profile = args.smithery_profile

    deployment_url = None

    # Try loading from source_file_path to get the connection URL
    source_path = server_info.get('source_file_path', '')
    if source_path:
        # Resolve relative path (source_file_path may start with ../)
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            resolved_path = os.path.normpath(os.path.join(script_dir, source_path))
            if os.path.exists(resolved_path):
                with open(resolved_path, 'r') as f:
                    source_data = json.load(f)
                server_data = source_data.get('server', {})
                connections = server_data.get('connections', [])
                if connections:
                    deployment_url = connections[0].get('deploymentUrl', '')
                if not deployment_url:
                    # Fallback to server-level deploymentUrl + /mcp
                    dep = server_data.get('deploymentUrl', '')
                    if dep:
                        deployment_url = dep.rstrip('/') + '/mcp'
        except Exception as e:
            print(f"⚠️ Warning: Could not load source file {source_path}: {e}")

    if not deployment_url:
        return None

    # Build the full URL with config, api_key, and profile
    config = {"debug": False}
    config_b64 = base64.b64encode(json.dumps(config).encode()).decode()
    server_url = f"{deployment_url}?config={config_b64}&api_key={api_key}&profile={profile}"
    return server_url


def create_agent_for_item(item, api_key=None, profile=None):
    """
    Create an OpenAI Agent for an item. 
    Supports both REAL MCP servers and VIRTUAL (LLM-generated) tools.
    """
    metadata = item.get('metadata', {})
    mcp_servers = metadata.get('mcp_servers', [])
    
    if not mcp_servers or not isinstance(mcp_servers, list):
        return None
    
    # --- CLIENT SETUP (Shared for both modes) ---
    client = AsyncClient(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    model = OpenAIResponsesModel(args.model_path, openai_client=client)

    # --- MODE 1: VIRTUAL TOOLS (Hallucinated) ---
    # You will need to add --virtual_tools to your args parser
    if hasattr(args, 'virtual_tools') and args.virtual_tools:
        ensure_question_generation_history_loaded()

        print(f"👻 Configuring Agent with VIRTUAL tools (Agent: {args.model_path}, VirtualTool: {args.model_path})...")
        virtual_backend = VirtualToolBackend(client, model_path=args.model_path)
        virtual_tool_funcs = []

        # Keep scenario context used by virtual tool simulation.
        question = metadata.get('question', '')
        tool_analysis = metadata.get('tool_analysis', '')
        workflow_analysis = metadata.get('cross_tool_workflow', '')

        target_tools = metadata.get('target_tools', [])
        expected_outputs_by_tool = {}
        for tt in (target_tools or []):
            if isinstance(tt, dict):
                tool_key = tt.get('tool', '')
                if tool_key:
                    expected_outputs_by_tool[tool_key] = tt.get('output', '')

        row_id_key = normalize_row_id_key(metadata.get('row_id'))
        matched_history = QUESTION_GEN_HISTORY_BY_ROW_ID.get(row_id_key) if row_id_key is not None else None
        if not isinstance(matched_history, list) or not matched_history:
            matched_history = item.get("messages", [])
        if not isinstance(matched_history, list) or not matched_history:
            matched_history = [{"role": "user", "content": question}] if question else []
        # Shared, per-item transcript of prior simulated tool calls.
        # Every virtual tool receives this same list for scenario continuity.
        tool_simulation_messages = []

        for server_info in mcp_servers:
            # Start with inline fields from the input file
            server_id = server_info.get('server_id', '')
            server_name = server_info.get('server_name', '')
            server_analysis = server_info.get('server_description', '')
            tools_list = server_info.get('tools', [])

            # Enrich from smithery directory if available (overrides inline values when found)
            if args.mcp_server_dir and server_id:
                smithery_path = os.path.join(args.mcp_server_dir, f"{server_id}.json")
                if os.path.exists(smithery_path):
                    with open(smithery_path) as sf:
                        smithery_data = json.load(sf)
                    smithery_tools = smithery_data.get('server', {}).get('tools', [])
                    tools_list = smithery_tools if smithery_tools else tools_list
                    server_analysis = smithery_data.get('analysis', server_analysis)
                    server_name = smithery_data.get('server', {}).get('displayName', server_name)

            for tool_def in tools_list:
                # Create the dynamic python function for this tool
                if server_analysis and server_name and tool_def['description']:
                    tool_def['description'] = f'''This tool comes from the MCP server: {server_name}. 
                    
                    An analysis of this server is as follows: {server_analysis}.
                    
                    This tool has the following functionality within the MCP server: {tool_def['description']}'''
                tool_raw_name = tool_def.get('name', '')
                expected_output = expected_outputs_by_tool.get(tool_raw_name, '')
                scenario_context = {
                    "conversation_history": matched_history,
                    "tool_simulation_messages": tool_simulation_messages,
                    "question": question,
                    "tool_analysis": tool_analysis,
                    "workflow_analysis": workflow_analysis,
                    "expected_output": expected_output,
                    "server_id": server_id,
                    "server_name": server_name,
                    "server_description": server_analysis,
                }
                v_tool = create_dynamic_virtual_tool(
                    tool_def,
                    virtual_backend,
                    scenario_context=scenario_context,
                )
                virtual_tool_funcs.append(v_tool)

        if not virtual_tool_funcs:
            print("❌ No tool definitions found in metadata for virtual generation.")
            return None

        # Return config with 'tools' instead of 'mcp_servers_list'
        return {
            "name": "OSS-Virtual-Assistant",
            "instructions": "You are a helpful assistant. Use the provided tools to answer the user query.",
            "model": model,
            "tools": virtual_tool_funcs, # <--- The Agent uses these directly
            "mcp_servers_list": [] # No real connections
        }

    # --- MODE 2: REAL MCP SERVERS (Existing Logic) ---
    else:
        mcp_servers_list = []
        for server_info in mcp_servers:
            server_url = None

            # Try existing format first: server_info sub-dict with python_sdk_url
            server_details = server_info.get('server_info', {})
            if server_details:
                server_url = construct_mcp_server_url(server_details, api_key, profile)

            # Fallback: step 1.1 format with source_file_path or server_id
            if not server_url:
                server_url = construct_mcp_url_from_source(server_info, api_key, profile)

            if server_url:
                safe_name = server_info.get('server_name', 'unknown').replace(' ', '-').lower()
                mcp_servers_list.append({
                    "name": safe_name,
                    "url": server_url,
                    "timeout": 600.0,
                    "sse_read_timeout": 600.0,
                    "terminate_on_close": False
                })

        if not mcp_servers_list:
            return None

        return {
            "name": "OSS-Assistant",
            "instructions": "You are a helpful assistant. Use the available tools.",
            "model": model,
            "mcp_servers_list": mcp_servers_list
        }

def qwen_compatible_system_prompt_generator(tools):
    """Generate a Qwen-compatible system prompt from tool specs.

    tools: iterable of FunctionTool-like objects with attributes:
      - name: str
      - description: str | None
      - params_json_schema: dict | None (JSON Schema for parameters)
    """
    import json as _json

    # Build function schema list expected by Qwen's NousFnCallPrompt
    function_schemas = []
    for tool in tools or []:
        name = getattr(tool, 'name', None) or ''
        description = getattr(tool, 'description', None) or ''
        params_schema = getattr(tool, 'params_json_schema', None) or {"type": "object", "properties": {}}

        function_schemas.append({
            "name": name,
            "description": description,
            "parameters": params_schema,
        })

    tool_descs_wrapped = [{"type": "function", "function": fs} for fs in function_schemas]
    tool_descs_str = "\n".join(_json.dumps(d, ensure_ascii=False) for d in tool_descs_wrapped)

    template = (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n{tool_descs}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        "{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n"
        "</tool_call>"
    )

    return template.format(tool_descs=tool_descs_str)

# Process a single item using agent inference
async def process_single_item_agent_async(item, api_key=None, profile=None):
    """Process a single item using agent inference (async version)"""
    # Get prompt ID for better error tracking
    prompt_id = item.get('metadata', {}).get('prompt_id', 'unknown')

    if args.enable_tool_hint:
        if "metadata" in item and "target_tools" in item["metadata"]:
            target_tools = item["metadata"].get('target_tools', "")
        else:
            target_tools = item.get("target_tools", "")
        tool_list = [tool.strip() for tool in target_tools.split(',')] 
        # remove contents before :: in tool_list
        tool_list = [tool.split('::')[1] if '::' in tool else tool for tool in tool_list]
        tool_list = [f"{tool}" for tool in tool_list]
        tool_list = ", ".join(tool_list)
        print(f"🔍 Tool list: {tool_list}")
    
    message = item["messages"]
    # remove the system prompt if it exists
    if message[0]['role'] == 'system':
        message = message[1:]
    
    # Extract the current user message (the last user message in the conversation)
    user_messages = [msg for msg in message if msg.get('role') == 'user']
    if user_messages:
        user_content = user_messages[-1]['content']
    else:
        raise ValueError("No user messages found")
    
    # Try to create agent for this item
    agent_config = None
    if args.agent:
        agent_config = create_agent_for_item(item, api_key, profile)
    
    if agent_config:
        try:
            # Use agent inference
            print(f"🚀 Running OpenAI agent inference for item {prompt_id}...")

            # Add tool hint if enabled
            if args.enable_tool_hint:
                # Get MCP server information for tool hint
                if tool_list:
                    tool_hint = f'\n\nWe need to use the following tools: {tool_list}.'
                else:
                    tool_hint = '\n\nWe need to use the provided tools.'
                user_content = user_content + tool_hint

            if args.enable_irrelevant_warning:
                user_content = user_content + '\n\nUse tools only if they are relevant. Otherwise, do not use them.'

            # Handle both single and multiple MCP servers
            server_configs = agent_config["mcp_servers_list"]
            mcp_servers = []
            server_contexts = []
            
            # Create a list to hold all MCP server context managers
            async def create_mcp_servers():
                mcp_servers = []
                server_contexts = []
                failed_servers = []

                for server_config in server_configs:
                    try:
                        mcp_server_context = MCPServerStreamableHttp(
                            name=server_config["name"],
                            params={
                                "url": server_config["url"],
                                "headers": {
                                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
                                },
                                "timeout": server_config.get("timeout", 600.0),
                                "sse_read_timeout": server_config.get("sse_read_timeout", 600.0),
                                "terminate_on_close": server_config.get("terminate_on_close", False)
                            },
                            client_session_timeout_seconds=args.timeout,
                        )

                        # Enter the context and collect both the server and its context
                        mcp_server = await mcp_server_context.__aenter__()
                        mcp_servers.append(mcp_server)
                        server_contexts.append(mcp_server_context)
                    except Exception as conn_error:
                        failed_servers.append(server_config["name"])
                        print(f"⚠️ Skipping MCP server '{server_config['name']}': {conn_error}")

                if failed_servers:
                    print(f"   Failed servers: {', '.join(failed_servers)} ({len(mcp_servers)}/{len(server_configs)} connected)")

                return mcp_servers, server_contexts
            
            # Create and manage multiple MCP servers
            try:
                if server_configs:
                    mcp_servers, server_contexts = await create_mcp_servers()

                # Fail if no MCP servers connected — we require real tool execution
                if not mcp_servers and not agent_config.get("tools"):
                    raise Exception(f"All {len(server_configs)} MCP server(s) failed to connect — cannot proceed without real tools")

                try:
                    # Create OpenAI Agent with multiple MCP servers
                    agent_kwargs = {
                    "name": agent_config["name"],
                    "instructions": agent_config["instructions"],
                    "model": agent_config["model"],}

                # conditionally add servers or tools based on what exists
                    if mcp_servers:
                        agent_kwargs["mcp_servers"] = mcp_servers
                    
                    if agent_config.get("tools"):
                        # This passes our Virtual Tool Functions to the Agent
                        agent_kwargs["tools"] = agent_config["tools"]
                    agent = Agent(**agent_kwargs)
                    run_context = RunContextWrapper(context=None)
                    
                    # print(f"🔍 User Query Passed to Agent: {user_content}")
                    # If this is a multi-turn conversation, populate the session with history
                    if len(message) > 1:
                        # Create a session for conversation management
                        # Use prompt_id as session identifier to maintain conversation history
                        session = SQLiteSession(f"conversation_{prompt_id}")

                        # Clear any existing session data first
                        await session.clear_session()
                        
                        # Add conversation history to session (all messages except the last user message)
                        history_items = []
                        for msg in message[:-1]:  # All messages except the last one
                            if msg['role'] == 'user':
                                history_items.append({"role": "user", "content": msg['content']})
                            elif msg['role'] == 'assistant':
                                history_items.append({"role": "assistant", "content": msg['content']})
                            elif msg['role'] == 'function':
                                # Convert function response to assistant message mentioning the function result
                                function_name = msg.get('name', 'unknown_function')
                                history_items.append({
                                    "role": "assistant", 
                                    "content": f"[Function {function_name} returned: {msg['content']}]"
                                })
                        
                        # Add history to session
                        if history_items:
                            await session.add_items(history_items)

                        # Run agent inference with session for automatic conversation management
                        result = await Runner.run(agent, input=user_content, session=session, max_turns=args.max_turns)
                    else:
                        result = await Runner.run(agent, input=user_content, max_turns=args.max_turns)

                    available_tools = await agent.get_all_tools(run_context)
                    system_prompt = qwen_compatible_system_prompt_generator(available_tools)

                    # Convert OpenAI Agent result to message format (this is the main conversation history)
                    all_messages = convert_openai_agent_result_to_messages(result, message, system_prompt)
                                    
                    if len(all_messages) > len(message):
                        # Check for MCP error patterns in the final assistant response
                        error_patterns = [
                            "[ERROR: Session terminated]",
                            "[ERROR: Failed to connect to MCP server",
                            "[ERROR:",
                        ]
                        final_assistant_msgs = [
                            m for m in all_messages if m.get('role') == 'assistant' and m.get('content')
                        ]
                        has_error_response = False
                        if final_assistant_msgs:
                            last_content = final_assistant_msgs[-1].get('content', '')
                            for pattern in error_patterns:
                                if pattern in last_content and len(last_content.strip()) < 200:
                                    has_error_response = True
                                    print(f"⚠️ Agent response for item {prompt_id} contains MCP error: {last_content.strip()}")
                                    break

                        if has_error_response:
                            raise Exception(f"Agent response is an MCP error: {last_content.strip()}")

                        tool_count = len(mcp_servers) if mcp_servers else len(agent_config.get("tools", []))
                        source_type = "MCP servers" if mcp_servers else "Virtual Tools"
                        print(f"✅ OpenAI agent inference completed for item {prompt_id} with {tool_count} {source_type}\n============================================================")
                        item['messages'] = all_messages
                    else:
                        print(f"⚠️ OpenAI agent inference returned empty response for item {prompt_id}\n============================================================")
                        raise Exception("Agent returned empty response")
                
                finally:
                    # Clean up all MCP server contexts
                    for server_context in reversed(server_contexts):
                        try:
                            await server_context.__aexit__(None, None, None)
                        except Exception as cleanup_error:
                            print(f"⚠️ Warning: Failed to cleanup MCP server context: {cleanup_error}")
            
            except Exception as server_creation_error:
                print(f"❌ Failed to create MCP servers: {server_creation_error}")
                raise
                
        except Exception as e:
            print(f"❌ OpenAI agent inference failed for item {prompt_id}: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
            if "async" in str(e).lower() or "context" in str(e).lower() or "sse" in str(e).lower():
                print(f"   🔍 This appears to be an async/context/MCP streaming error")
    
            # Re-raise the exception to trigger fallback instead of returning empty content
            raise e
    else:
        # If no agent could be created, raise an exception to trigger fallback
        if args.agent:
            raise ValueError("Failed to create agent for this item")
        else:
            raise ValueError("No agent specified")
    
    return item

def extract_message_text(content):
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        text_chunks = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_chunks.append(part.get("text", ""))
        return "".join(text_chunks)
    return str(content)


def normalize_messages_for_completion(messages):
    if not messages:
        raise ValueError("No messages found")

    input_messages = messages
    if input_messages[0].get("role") == "system":
        input_messages = input_messages[1:]

    user_messages = [msg for msg in input_messages if msg.get("role") == "user"]
    if not user_messages:
        raise ValueError("No user messages found")

    latest_user_content = user_messages[-1]["content"]
    if input_messages[-1].get("role") != "user":
        raise ValueError("Last message is not a user message")

    return input_messages[:-1] + [{"role": "user", "content": latest_user_content}]


async def request_completion_async(messages, client):
    payload = {
        "model": args.model_path,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "messages": messages,
    }
    extra_body = {
        "parallel_tool_calls": args.parallel_function_calls,
        "reasoning": {"effort": args.reasoning_effort},
    }
    if args.max_tokens is not None and args.max_tokens > 0:
        payload["max_tokens"] = args.max_tokens
    if extra_body:
        payload["extra_body"] = extra_body

    for attempt in range(args.max_retries):
        try:
            completion = await client.chat.completions.create(**payload)
            return extract_message_text(completion.choices[0].message.content).strip()
        except Exception as e:
            print(f"Request attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2 ** attempt)

    return ""


async def process_single_item_direct_async(item, client):
    prompt_id = item.get("metadata", {}).get("prompt_id", "unknown")
    input_messages = normalize_messages_for_completion(item["messages"])

    print(f"🔄 Using direct API for item {prompt_id}...")
    response = await request_completion_async(input_messages, client)
    if response is None:
        response = ""

    item["messages"] = input_messages + [{"role": "assistant", "content": response}]
    return item

# Function to sort dataset by row_id from metadata
def sort_dataset_by_row_id(dataset):
    """Sort dataset by row_id from metadata, handling missing row_ids gracefully"""
    def get_sort_key(item):
        metadata = item.get('metadata', {})
        row_id = metadata.get('row_id')
        if row_id is not None:
            try:
                return int(row_id)
            except (ValueError, TypeError):
                # If row_id can't be converted to int, use as string
                return float('inf'), str(row_id)
        else:
            # Items without row_id go to the end
            return float('inf'), ''
    
    return sorted(dataset, key=get_sort_key)

# Function to add generation config to metadata
def add_generation_config_to_metadata(dataset, model_short_name, generation_params):
    """Add synthetic data generation config to each item's metadata"""
    config_entry = {
        "model": model_short_name,
        "generation_params": generation_params,
        "timestamp": int(time())
    }
    
    for item in dataset:
        if "metadata" not in item:
            item["metadata"] = {}
        
        if "synthetic_data_gen_configs" not in item["metadata"]:
            item["metadata"]["synthetic_data_gen_configs"] = []
        
        item["metadata"]["synthetic_data_gen_configs"].append(config_entry)
    
    return dataset

def checkpoint_file_path(index, checkpoint_dir):
    return os.path.join(checkpoint_dir, f"{index:08d}.json")


def save_item_checkpoint(index, item, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = checkpoint_file_path(index, checkpoint_dir)
    temp_path = f"{checkpoint_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")
    os.replace(temp_path, checkpoint_path)


def load_item_checkpoints(processed_dataset, checkpoint_dir):
    completed_indices = set()
    if not os.path.isdir(checkpoint_dir):
        return completed_indices

    for file_name in os.listdir(checkpoint_dir):
        match = re.fullmatch(r"(\d+)\.json", file_name)
        if match is None:
            continue

        index = int(match.group(1))
        if index < 0 or index >= len(processed_dataset):
            continue

        file_path = os.path.join(checkpoint_dir, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                processed_dataset[index] = json.load(f)
            completed_indices.add(index)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Failed to load checkpoint {file_path}: {e}. Reprocessing index {index}.")

    return completed_indices


def get_input_base_name(input_file):
    base_name = input_file[: input_file.rfind(".")]
    if base_name.endswith("_4prepared"):
        return base_name[:-10]
    if base_name.endswith("_prepared"):
        return base_name[:-9]
    return base_name


def resolve_processing_range(total_rows):
    if total_rows <= 0:
        raise ValueError(f"Input dataset is empty: {args.input_file}")
    if args.start_idx >= total_rows:
        raise ValueError(
            f"--start_idx ({args.start_idx}) must be smaller than dataset size ({total_rows})."
        )

    start_idx = args.start_idx
    requested_end_idx = total_rows if args.batch_size is None else args.start_idx + args.batch_size
    end_idx = min(requested_end_idx, total_rows)
    return start_idx, requested_end_idx, end_idx


def build_output_paths(base_name, start_idx, end_idx, trial_idx=None):
    trial_suffix = f"{trial_idx}" if trial_idx is not None else ""
    range_mode = args.batch_size is not None or args.start_idx != 0
    if range_mode:
        saved_file = f"{base_name}_{config_str}_results{trial_suffix}_{start_idx}_{end_idx}.jsonl"
        checkpoint_dir = f"{base_name}_{config_str}_results{trial_suffix}_checkpoints_{start_idx}_{end_idx}"
    else:
        saved_file = f"{base_name}_{config_str}_results{trial_suffix}.jsonl"
        checkpoint_dir = f"{base_name}_{config_str}_results{trial_suffix}_checkpoints"
    return saved_file, checkpoint_dir


def build_generation_params(max_workers):
    return {
        "base_url": args.base_url,
        "model_path": args.model_path,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "num_trials": args.num_trials,
        "step": args.step,
        "agent": args.agent,
        "timeout": args.timeout,
        "max_workers": max_workers,
        "parallel_function_calls": args.parallel_function_calls,
        "reasoning_effort": args.reasoning_effort,
    }


def build_error_item(item, error_text):
    fallback_item = copy.deepcopy(item)
    original_messages = fallback_item.get("messages", [])
    fallback_item["messages"] = original_messages + [
        {"role": "assistant", "content": f"[ERROR: {error_text}]"}
    ]
    return fallback_item


async def process_index_async(index, processed_dataset, direct_client, semaphore):
    async with semaphore:
        api_key, profile = get_api_key_for_worker(index)
        current_item = copy.deepcopy(processed_dataset[index])
        prompt_id = current_item.get("metadata", {}).get("prompt_id", f"item_{index}")

        try:
            if args.agent:
                processed_item = await asyncio.wait_for(
                    process_single_item_agent_async(current_item, api_key, profile),
                    timeout=args.timeout,
                )
            else:
                processed_item = await asyncio.wait_for(
                    process_single_item_direct_async(current_item, direct_client),
                    timeout=args.timeout,
                )
            print(f"✅ Completed item {prompt_id} (index {index})")
            return index, processed_item
        except Exception as e:
            if isinstance(e, asyncio.TimeoutError):
                error_text = f"Timed out after {args.timeout}s"
            else:
                error_text = str(e)
            print(f"❌ Failed item {prompt_id} (index {index}): {error_text}")
            return index, build_error_item(current_item, error_text)


async def generate_and_update(dataset, direct_client, checkpoint_dir):
    processed_dataset = copy.deepcopy(dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)

    completed_indices = load_item_checkpoints(processed_dataset, checkpoint_dir)
    if completed_indices:
        print(
            f"Loaded {len(completed_indices)} completed item checkpoints from {checkpoint_dir}."
        )

    pending_indices = [idx for idx in range(len(processed_dataset)) if idx not in completed_indices]
    max_workers = args.max_workers or (len(smithery_api_pool) if smithery_api_pool else 8)

    if not pending_indices:
        print("No remaining items to process.")
    else:
        print(f"Processing {len(pending_indices)} items with max concurrency {max_workers}.")
        semaphore = asyncio.Semaphore(max_workers)
        tasks = [
            asyncio.create_task(process_index_async(idx, processed_dataset, direct_client, semaphore))
            for idx in pending_indices
        ]

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating completions"):
            index, processed_item = await task
            processed_dataset[index] = processed_item
            save_item_checkpoint(index, processed_item, checkpoint_dir)

    generation_params = build_generation_params(max_workers)
    processed_dataset = add_generation_config_to_metadata(
        processed_dataset,
        model_abbreviation,
        generation_params,
    )
    return sort_dataset_by_row_id(processed_dataset)


async def run_trial(target_dataset, saved_file, checkpoint_dir):
    direct_client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    try:
        updated_dataset = await generate_and_update(target_dataset, direct_client, checkpoint_dir)
        save_dataset(updated_dataset, saved_file, convert_to_jsonl=True)
        if os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        print(f"Final dataset saved to {saved_file}.")
    finally:
        await direct_client.close()


async def main():
    if args.num_trials <= 0:
        raise ValueError("--num_trials must be a positive integer.")

    api_pool = load_and_validate_smithery_api_pool(args.smithery_api_pool)
    pool_size = len(api_pool) if api_pool else 0
    effective_workers = args.max_workers or (pool_size if pool_size > 0 else 8)

    print("=" * 50)
    print("🚀 ASYNC PROCESSING CONFIGURATION")
    print("=" * 50)
    print(f"Workers: {effective_workers}")
    print(f"API pool size: {pool_size}")
    print(f"Timeout per item: {args.timeout} seconds")
    print(f"Checkpointing: One JSON file per completed item")
    print(f"Endpoint: {args.base_url}")
    print(f"Mode: {'Agent' if args.agent else 'Direct API'}")
    print("=" * 50)

    dataset = load_dataset_from_file(INPUT_FILE_NAME)
    if not isinstance(dataset, list):
        dataset = [dataset]

    total_rows = len(dataset)
    start_idx, requested_end_idx, end_idx = resolve_processing_range(total_rows)
    target_dataset = dataset[start_idx:end_idx]

    base_name = get_input_base_name(args.input_file)
    print(
        f"Dataset rows: {total_rows}. Requested range: "
        f"[{start_idx}, {requested_end_idx}). Effective range: [{start_idx}, {end_idx})."
    )
    print(f"Processing {len(target_dataset)} rows.")

    if args.num_trials == 1:
        saved_file, checkpoint_dir = build_output_paths(base_name, start_idx, end_idx)
        print(f"Output file: {saved_file}")
        print(f"Checkpoint dir: {checkpoint_dir}")
        await run_trial(target_dataset, saved_file, checkpoint_dir)
    else:
        for trial_idx in range(args.num_trials):
            saved_file, checkpoint_dir = build_output_paths(base_name, start_idx, end_idx, trial_idx=trial_idx)
            print(f"Trial {trial_idx}: output={saved_file}")
            print(f"Trial {trial_idx}: checkpoints={checkpoint_dir}")
            await run_trial(target_dataset, saved_file, checkpoint_dir)

    print("🎯 Program execution completed.")


if __name__ == "__main__":
    asyncio.run(main())
