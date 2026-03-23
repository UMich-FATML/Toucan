# O*NET Smoke Test Through Step 3.1

This runbook tests the minimal O*NET task-first pipeline in `datagen-v2` through step 3.1 on a fixed 100-prompt smoke sample.

It is intentionally opinionated:
- Use `100` prompts.
- Use the O*NET path only.
- Use `--job_name smoke_test`.
- Run step 1.2 with `structured_completions_endpoint.py` directly.
- Do not pass `--start_idx` or `--batch_size` in step 1.2, so the output is a canonical parent `*_results.jsonl`.
- Run step 3.1 with `completion_openai_agent.py` directly in `--virtual_tools` mode.
- Use either:
  - a vLLM-compatible endpoint such as `http://fs-mbz-gpu-758:8000/v1` with API key `EMPTY`, or
  - OpenRouter at `https://openrouter.ai/api/v1` with the API key from `OPENROUTER_API_KEY`.

That last point matters because step 3.1 looks for a sibling parent `*_results.jsonl` when it reconstructs question-generation history.

## Prerequisites

- Run commands from the `datagen-v2` directory:
  ```bash
  cd /mnt/weka/home/yuekai.sun/Toucan/datagen-v2
  ```
- Activate the Python environment:
  ```bash
  conda activate toucan
  ```
- Ensure your selected endpoint is reachable for step 1.2 and step 3.1.
- Ensure `smithery_api_pool.json` is populated for step 3.1.
- Ensure `../mcp_servers/smithery_mcp_servers_0210` exists for step 1.1 metadata loading and step 3.1 virtual-tool enrichment.

Optional preflight checks:

```bash
python -m py_compile \
  step1.1_gen_questions_from_onet_tasks.py \
  structured_completions_endpoint.py \
  step1.3_process_onet_completion.py \
  step2_validate_and_convert.py \
  completion_openai_agent.py \
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

## Step 1.1

Generate the O*NET task-first prompt set:

```bash
python step1.1_gen_prompts_from_onet_tasks.py \
  --num_tasks 3 \
  --total_prompts 100 \
  --self_contained \
  --no_refs \
  --job_name smoke
```

Capture the prepared file:

```bash
STEP11="$(ls -t ../data/smoke/onet_tasks_3_tasks_100_*_prepared.jsonl | head -n 1)"
echo "$STEP11"
wc -l "$STEP11"
```

Expected result:
- One `*_prepared.jsonl` file.
- `wc -l` reports `100`.

## Step 1.2

Run the O*NET completion step directly through the structured endpoint:

```bash
python structured_completions_endpoint.py \
  --input_file "$STEP11" \
  --model_name "$SMOKE_MODEL" \
  --base_url "$SMOKE_BASE_URL" \
  --api_key "$SMOKE_API_KEY" \
  --output_schema_file prompts/genq_from_onet_tasks_output_schema.json \
  --step 1.2_onet_smoke
```

Resolve the canonical output path and enforce the parent filename if needed:

```bash
STEP12="$(ls -t ../data/smoke/onet_tasks_3_tasks_100_*_${SMOKE_MODEL}_results.jsonl 2>/dev/null | head -n 1 || true)"
if [ -z "$STEP12" ]; then
  RANGED="$(ls -t ../data/smoke/onet_tasks_3_tasks_100_*_${SMOKE_MODEL}_results_*_*.jsonl | head -n 1)"
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
- `wc -l` reports `100`.

## Step 1.3

Process and sanitize the O*NET completions:

```bash
python step1.3_process_onet_completion.py --input_file "$STEP12"
```

Capture the sanitized output:

```bash
STEP13="$(ls -t "$(dirname "$STEP12")"/processed/*_3sanitized.jsonl | head -n 1)"
echo "$STEP13"
wc -l "$STEP13"
```

Expected result:
- A `processed/*_3sanitized.jsonl` file exists.
- `wc -l` reports `100` for the current smoke-test configuration.

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
- `wc -l` reports `100`.

Optional artifact:
- `*_answer_key.jsonl`

## Step 3.1

Generate agent trajectories with the direct virtual-tools invocation:

```bash
python completion_openai_agent.py \
  --input_file "$STEP2" \
  --model_path "$SMOKE_MODEL" \
  --base_url "$SMOKE_BASE_URL" \
  --api_key "$SMOKE_API_KEY" \
  --step 3.1 \
  --agent openai_agent \
  --virtual_tools \
  --mcp_server_dir ../mcp_servers/smithery_mcp_servers_0210 \
  --max_workers 80 \
  --timeout 1200
```

Capture the latest step 3.1 result:

```bash
STEP31="$(ls -t "$(dirname "$STEP2")"/*_${SMOKE_MODEL}_high_pfc_results.jsonl | head -n 1)"
echo "$STEP31"
wc -l "$STEP31"
rg -c "\\[ERROR:" "$STEP31"
```

Expected result:
- A step 3.1 `*_high_pfc_results.jsonl` file exists.
- `wc -l` reports `100`.
- The error count may be non-zero; record it rather than treating the smoke test as automatically failed.
- In the latest reference run, step 3.1 completed with `100` rows and `2` terminal `[ERROR: ...]` rows, including one timeout.

## What To Verify

- Step 1.1 writes `100` prepared rows.
- Step 1.2 writes a canonical parent `*_results.jsonl`.
- Step 1.3 writes `*_3sanitized.jsonl`.
- Step 2 writes `*_validated_prepared.jsonl`.
- Step 3.1 writes `*_high_pfc_results.jsonl`.
- Step 3.1 logs successful question-history lookup against the parent `*_results.jsonl`.

## Failure Triage

- If step 1.1 fails immediately, verify:
  - `tasks_to_smithery_servers.jsonl`
  - `onet_db_30_1_text/Occupation Data.txt`
  - `../mcp_servers/smithery_mcp_servers_0210`
- If step 1.2 fails immediately, verify:
  - `SMOKE_BASE_URL`
  - `SMOKE_API_KEY`
  - `OPENROUTER_API_KEY` if you are using OpenRouter
  - `prompts/genq_from_onet_tasks_output_schema.json`
- If step 1.3 cannot find the input file, verify that step 1.2 wrote the canonical non-sharded output.
- If step 3.1 fails immediately, verify:
  - `SMOKE_BASE_URL`
  - `SMOKE_API_KEY`
  - `OPENROUTER_API_KEY` if you are using OpenRouter
  - `smithery_api_pool.json`
  - `../mcp_servers/smithery_mcp_servers_0210`
- If step 3.1 cannot load question history, verify the parent step 1.2 file is named `*_results.jsonl` and lives one directory above `processed/`.
- If step 3.1 finishes with some `[ERROR: ...]` rows, keep the output file and record the count. For a smoke test, partial completion is still informative.
