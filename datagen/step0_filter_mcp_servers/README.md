# Step 0: Filter MCP Servers by Tool Quality

This step tests each validated MCP server by having an agent call its tools and report pass/fail quality results. The output is used to filter the server pool before expensive data generation steps.

## Overview

```
prepare_server_quality_check_prompts.py
    ↓  (*_prepared.jsonl — one item per server)
server_quality_check_agent.sh  (calls completion_openai_agent.py)
    ↓  (*_results.jsonl — full agent trajectories, kept for inspection)
process_server_quality_check.py
    ↓  (server_quality_results.jsonl — per-server quality report)
    ↓
[optional] prepare_tool_retry_prompts.py  — retry failed/specific tools
    ↓  (*_prepared.jsonl — one item per tool)
tool_retry_agent.sh
    ↓  (*_results.jsonl)
process_tool_retry_results.py
    ↓  (server_quality_results_merged.jsonl — final merged output)
```

**Run all scripts from the `datagen/` directory** (or any directory — the scripts resolve paths relative to their own location).

---

## Step 1: Prepare prompts

```bash
python step0_filter_mcp_servers/prepare_server_quality_check_prompts.py \
    --server_dir ../mcp_servers/smithery_mcp_servers_0210/ \
    --output_file ../data/server_quality_check_prepared.jsonl
```

Options:
- `--max_servers N` — cap for testing (e.g. `--max_servers 5`)

---

## Step 2: Run agent

```bash
export OPENROUTER_API_KEY=<your_key>

bash step0_filter_mcp_servers/server_quality_check_agent.sh \
    ../data/server_quality_check_prepared.jsonl \
    qwen/qwen3-30b-a3b-thinking-2507
```

Output file: `../data/server_quality_check_prepared_<model>_results.jsonl`

Optional env vars: `MAX_WORKERS` (default 4), `TIMEOUT` (default 300s), `MAX_TURNS` (default 20).

---

## Step 3: Post-process results

```bash
python step0_filter_mcp_servers/process_server_quality_check.py \
    --input_file ../data/server_quality_check_prepared_<model>_results.jsonl \
    --output_file ../data/server_quality_results.jsonl \
    --server_dir ../mcp_servers/smithery_mcp_servers_0210/
```

The intermediate `_results.jsonl` is preserved for trajectory inspection.

---

## Step 4 (optional): Retry failed tools

For servers where the agent hit errors (timeout, max_turns, etc.), retry each tool individually:

```bash
# Prepare per-tool prompts for servers with retryable agent errors
python step0_filter_mcp_servers/prepare_tool_retry_prompts.py \
    --quality_results_file ../data/server_quality_results.jsonl \
    --server_dir ../mcp_servers/smithery_mcp_servers_0210/ \
    --output_file ../data/tool_retry_prepared.jsonl

# Run agent (shorter timeout, fewer turns)
bash step0_filter_mcp_servers/tool_retry_agent.sh \
    ../data/tool_retry_prepared.jsonl \
    qwen/qwen3-30b-a3b-thinking-2507

# Merge retry results back in
python step0_filter_mcp_servers/process_tool_retry_results.py \
    --input_file ../data/tool_retry_prepared_<model>_results.jsonl \
    --quality_results_file ../data/server_quality_results.jsonl \
    --output_file ../data/server_quality_results_merged.jsonl
```

By default, `context_length`, `mcp_invalid_response`, `mcp_method_not_found`, and `timeout` errors are skipped (not retried). Pass `--skip_error_types` to override.

### Tool-level retry (e.g. connection failures)

To retry specific tools whose result reasoning matches a pattern:

```bash
python step0_filter_mcp_servers/prepare_tool_retry_prompts.py \
    --quality_results_file ../data/server_quality_results_merged.jsonl \
    --server_dir ../mcp_servers/smithery_mcp_servers_0210/ \
    --output_file ../data/tool_retry2_prepared.jsonl \
    --mode tool \
    --tool_error_patterns "mcp_connection_failed"
```

---

## Output format

Each line of `server_quality_results*.jsonl`:

```json
{
  "server_id": "<uuid>",
  "server_name": "<displayName>",
  "qualified_name": "<qualifiedName>",
  "agent_error": false,
  "agent_error_type": null,
  "agent_error_msg": null,
  "tool_results": [
    {"tool_name": "search", "quality": "pass", "reasoning": "Returned relevant results for test query"},
    {"tool_name": "write_file", "quality": "fail", "reasoning": "403 permission denied on all attempts"}
  ],
  "metadata": {
    "server_info": { /* full original server JSON */ },
    "model": "<model>",
    "timestamp": 1234567890,
    "num_tools_checked": 2,
    "num_tools_passed": 1
  }
}
```

`agent_error: true` means the agent framework itself failed (not the server tools). These entries have `tool_results` populated with fallback fail entries.

### Agent error types

| Type | Meaning |
|---|---|
| `timeout` | Worker timed out |
| `max_turns_exceeded` | Agent hit `--max_turns` limit |
| `context_length` | Prompt exceeded model token limit |
| `mcp_connection_failed` | All MCP server connections failed |
| `mcp_invalid_response` | Server returned non-MCP-compliant response |
| `mcp_method_not_found` | Server doesn't implement the requested method |
| `tool_not_found` | Named tool missing from agent tool list |
| `api_error` | HTTP error from LLM API |
| `other` | Unclassified error |
