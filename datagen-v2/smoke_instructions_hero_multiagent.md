# Hero Mini Multi-Agent Smoke Test Through Step 3.1m

This runbook tests the hero-mini **multi-turn** pipeline in `datagen-v2` through step 3.1m.
The default `--no_refs` prompt template already instructs the LLM to generate withheld
information. The withheld output schema at step 1.2 enforces `withheld_info` and
`target_followup_questions` as required fields. The multi-agent Student-User loop then
exercises clarification turns.

Choose a task mode before starting:

| Mode | `--num_tasks` | `--max_per_occupation` | Approximate prompts |
|------|---------------|------------------------|---------------------|
| 2-task | 2 | 231 | 231 per occupation |
| 3-task | 3 | 1540 | 1540 per occupation |

It is intentionally opinionated:
- Use the hero-mini path (`step1.1_hero_mini_gen.py`).
- Use `--job_name smoke_test_multiagent`.
- Use `--no_refs` (the default template already generates withheld info; the withheld output schema at step 1.2 enforces it).
- Run step 1.2 with `structured_completions_endpoint.py` directly, using the withheld output schema.
- Do not pass `--start_idx` or `--batch_size` in step 1.2, so the output is a canonical parent `*_results.jsonl`.
- Run step 1.3 with `--disable_sanitize`.
- Run step 3.1m with `completion_multiagent.py` directly in `--virtual_tools` mode.
- Use either:
  - a vLLM-compatible endpoint such as `http://fs-mbz-gpu-758:8000/v1` with API key `EMPTY`, or
  - OpenRouter at `https://openrouter.ai/api/v1` with the API key from `OPENROUTER_API_KEY`.

Step 3.1m looks for a sibling parent `*_results.jsonl` when it reconstructs question-generation history.

## Prerequisites

- Run commands from the `datagen-v2` directory:
  ```bash
  cd /mnt/weka/home/yuekai.sun/Toucan/datagen-v2
  ```
- Activate the Python environment:
  ```bash
  conda activate toucan
  ```
- Ensure your selected endpoint is reachable for step 1.2 and step 3.1m.
- Ensure `smithery_api_pool.json` is populated for step 3.1m.
- Ensure `../mcp_servers/smithery_mcp_servers_0210` exists for step 1.1 metadata loading and step 3.1m virtual-tool enrichment.

Optional preflight checks:

```bash
python -m py_compile \
  step1.1_hero_mini_gen.py \
  structured_completions_endpoint.py \
  step1.3_process_onet_completion.py \
  step2_validate_and_convert.py \
  completion_multiagent.py \
  virtual_tools.py \
  utils.py
```

Set the smoke-test runtime variables for one of the supported providers.

For vLLM:

```bash
export SMOKE_BASE_URL="http://fs-mbz-gpu-758:8000/v1"
export SMOKE_API_KEY="EMPTY"
export SMOKE_MODEL="moonshotai/kimi-k2.5"
```

For OpenRouter:

```bash
export SMOKE_BASE_URL="https://openrouter.ai/api/v1"
export SMOKE_API_KEY="${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set}"
export SMOKE_MODEL="moonshotai/kimi-k2.5"
```

`SMOKE_MODEL` must be a model name that your selected provider accepts.

Set the task mode:

For 2-task mode:

```bash
export SMOKE_NUM_TASKS=2
export SMOKE_MAX_PER_OCC=231
```

For 3-task mode:

```bash
export SMOKE_NUM_TASKS=3
export SMOKE_MAX_PER_OCC=1540
```

## Step 1.1

Generate the hero-mini prompt set:

```bash
python step1.1_hero_mini_gen.py \
  --no_refs \
  --num_tasks "$SMOKE_NUM_TASKS" \
  --max_per_occupation "$SMOKE_MAX_PER_OCC" \
  --job_name smoke_test_multiagent
```

Capture the prepared file:

```bash
STEP11="$(ls -t ../data/smoke_test_multiagent/*_prepared.jsonl | head -n 1)"
echo "$STEP11"
wc -l "$STEP11"
```

Expected result:
- One `*_prepared.jsonl` file.
- `wc -l` reports the number of prepared rows (up to `SMOKE_MAX_PER_OCC` per selected occupation).

## Step 1.2

Run the completion step directly through the structured endpoint.
Note: use the **withheld** output schema to enforce `withheld_info` and `target_followup_questions` as required fields.

```bash
python structured_completions_endpoint.py \
  --input_file "$STEP11" \
  --model_name "$SMOKE_MODEL" \
  --base_url "$SMOKE_BASE_URL" \
  --api_key "$SMOKE_API_KEY" \
  --output_schema_file prompts/genq_from_onet_tasks_withheld_output_schema.json \
  --step 1.2_hero_multiagent_smoke
```

Resolve the canonical output path and enforce the parent filename if needed:

```bash
STEP12="$(ls -t ../data/smoke_test_multiagent/*_${SMOKE_MODEL}_results.jsonl 2>/dev/null | head -n 1 || true)"
if [ -z "$STEP12" ]; then
  RANGED="$(ls -t ../data/smoke_test_multiagent/*_${SMOKE_MODEL}_results_*_*.jsonl | head -n 1)"
  PREFIX="$(echo "$RANGED" | sed -E 's/_results_[0-9]+_[0-9]+\.jsonl$//')"
  STEP12="${PREFIX}_results.jsonl"
  mv -f "$RANGED" "$STEP12"
fi
echo "$STEP12"
wc -l "$STEP12"
```

Expected result:
- Output file name ends with `*_results.jsonl`.
- It must not end with `*_results_<start>_<end>.jsonl`.
- `wc -l` matches the row count from step 1.1.

## Step 1.3

Process the completions (with sanitization disabled):

```bash
python step1.3_process_onet_completion.py --input_file "$STEP12" --disable_sanitize
```

Capture the output:

```bash
STEP13="$(ls -t "$(dirname "$STEP12")"/processed/*_3sanitized.jsonl | head -n 1)"
echo "$STEP13"
wc -l "$STEP13"
```

Expected result:
- A `processed/*_3sanitized.jsonl` file exists.
- `wc -l` matches the row count from step 1.1.

Quick check that `withheld_info` survived extraction:

```bash
python -c "
import json, sys
with open('$STEP13') as f:
    items = [json.loads(l) for l in f if l.strip()]
has_withheld = sum(1 for it in items if it.get('metadata',{}).get('withheld_info'))
print(f'Items with withheld_info: {has_withheld}/{len(items)}')
if has_withheld == 0:
    print('WARNING: no withheld_info found -- check step 1.1 used --withheld')
    sys.exit(1)
"
```

## Step 2

Convert the sanitized file into the agent-ready prepared format:

```bash
python step2_validate_and_convert.py --input_file "$STEP13"
```

Capture the validated prepared file:

```bash
STEP2="${STEP13%_3sanitized.jsonl}_validated_prepared.jsonl"
echo "$STEP2"
wc -l "$STEP2"
```

Expected result:
- A `*_validated_prepared.jsonl` file exists.
- `wc -l` matches the row count from step 1.1.

Optional artifact:
- `*_answer_key.jsonl`

## Step 3.1m

Generate multi-agent trajectories with virtual tools:

```bash
python completion_multiagent.py \
  --input_file "$STEP2" \
  --model_path "$SMOKE_MODEL" \
  --base_url "$SMOKE_BASE_URL" \
  --api_key "$SMOKE_API_KEY" \
  --step 3.1m \
  --agent openai_agent \
  --virtual_tools \
  --user_max_turns 10 \
  --mcp_server_dir ../mcp_servers/smithery_mcp_servers_0210 \
  --max_workers 100 \
  --timeout 1800
```

Capture the latest step 3.1m result:

```bash
STEP31m="$(ls -t "$(dirname "$STEP2")"/*_multiagent_pfc_results.jsonl | head -n 1)"
echo "$STEP31m"
wc -l "$STEP31m"
rg -c "\\[ERROR:" "$STEP31m"
```

Expected result:
- A step 3.1m `*_multiagent_pfc_results.jsonl` file exists.
- `wc -l` matches the row count from step 1.1.
- The error count may be non-zero; record it rather than treating the smoke test as automatically failed.

Quick multi-turn sanity check:

```bash
python -c "
import json
with open('$STEP31m') as f:
    items = [json.loads(l) for l in f if l.strip()]
for it in items[:3]:
    msgs = it.get('messages', [])
    user_turns = sum(1 for m in msgs if m.get('role') == 'user')
    asst_turns = sum(1 for m in msgs if m.get('role') == 'assistant')
    fn_turns = sum(1 for m in msgs if m.get('role') == 'function')
    pid = it.get('metadata',{}).get('prompt_id','?')
    print(f'{pid}: {len(msgs)} msgs (user={user_turns}, asst={asst_turns}, fn={fn_turns})')
"
```

## What To Verify

- Step 1.1 writes prepared rows (one per selected occupation).
- Step 1.2 writes a canonical parent `*_results.jsonl`.
- Step 1.3 writes `*_3sanitized.jsonl` (with sanitization disabled) and preserves `withheld_info`.
- Step 2 writes `*_validated_prepared.jsonl` with `withheld_info` intact.
- Step 3.1m writes `*_multiagent_pfc_results.jsonl`.
- Step 3.1m conversations contain multiple user turns (not just one).
- Step 3.1m logs successful question-history lookup against the parent `*_results.jsonl`.

## Failure Triage

- If step 1.1 fails immediately, verify:
  - `tasks_to_smithery_servers.jsonl`
  - `onet_db_30_1_text/Occupation Data.txt`
  - `../mcp_servers/smithery_mcp_servers_0210`
  - `../data/selected_ai_occupations.json`
- If step 1.2 fails immediately, verify:
  - `SMOKE_BASE_URL`
  - `SMOKE_API_KEY`
  - `OPENROUTER_API_KEY` if you are using OpenRouter
  - `prompts/genq_from_onet_tasks_withheld_output_schema.json` exists
- If step 1.3 cannot find the input file, verify that step 1.2 wrote the canonical non-sharded output.
- If withheld_info is missing after step 1.3, verify step 1.2 used `genq_from_onet_tasks_withheld_output_schema.json`.
- If step 3.1m fails immediately, verify:
  - `SMOKE_BASE_URL`
  - `SMOKE_API_KEY`
  - `OPENROUTER_API_KEY` if you are using OpenRouter
  - `smithery_api_pool.json`
  - `../mcp_servers/smithery_mcp_servers_0210`
  - `prompts/user.md` and `prompts/student.md` exist
- If step 3.1m cannot load question history, verify the parent step 1.2 file is named `*_results.jsonl` and lives one directory above `processed/`.
- If step 3.1m finishes with some `[ERROR: ...]` rows, keep the output file and record the count. For a smoke test, partial completion is still informative.
- If conversations only have a single turn, check that `withheld_info` is present in metadata and that the User agent prompt template (`prompts/user.md`) includes `{WITHHELD_INFO}`.
