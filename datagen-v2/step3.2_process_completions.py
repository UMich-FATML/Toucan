import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from transformers import AutoTokenizer

import process_entry as process_entry_module


DEFAULT_TOKENIZER_DIR = "/mnt/weka/shrd/k2m/yuekai.sun/bbq-chat-template/bbq-mixed-tool-format"
MAX_ROWS_PER_SHARD = 100_000
TOOL_BLOCK_RE = re.compile(r"<tools>(.*?)</tools>", re.DOTALL)
SHARDED_RESULTS_FILENAME_RE = re.compile(r"^(?P<prefix>.+)_results_(?P<start>\d+)_(?P<end>\d+)\.jsonl$")
CHECKPOINT_DIRNAME_RE = re.compile(r"^(?P<prefix>.+)_results_checkpoints_(?P<start>\d+)_(?P<end>\d+)$")
CHECKPOINT_FILENAME_RE = re.compile(r"^(?P<index>\d+)\.json$")
SYSTEM_BOILERPLATE_LINES = (
    "# Tools",
    "You may call one or more functions to assist with the user query.",
    "You are provided with function signatures within  XML tags:",
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:",
    "<tool_call>",
    '{"name": <function-name>, "arguments": <args-json-object>}',
    "</tool_call>",
)


@dataclass
class FailureRecord:
    line_no: int
    stage: str
    details: str


@dataclass
class RunStats:
    total_rows: int = 0
    rows_written: int = 0
    json_failures: int = 0
    normalization_failures: int = 0
    validation_failures: int = 0
    skipped_unknown_roles: int = 0
    bucket_counts: dict[str, int] = field(default_factory=dict)
    failure_examples: list[FailureRecord] = field(default_factory=list)

    def add_failure(self, line_no: int, stage: str, details: str, max_examples: int = 10) -> None:
        if len(self.failure_examples) < max_examples:
            self.failure_examples.append(FailureRecord(line_no=line_no, stage=stage, details=details))


@dataclass(frozen=True)
class ResultRangeSource:
    start_idx: int
    end_idx: int
    path: str
    source_type: str


@dataclass
class MergeRecoveryStats:
    line_count: int = 0
    existing_shard_count: int = 0
    recovered_shard_count: int = 0
    checkpoint_dir_count: int = 0
    skipped_checkpoint_file_count: int = 0
    recovered_ranges: list[str] = field(default_factory=list)
    gap_ranges: list[str] = field(default_factory=list)


class ShardWriter:
    def __init__(self, bucket_dir: Path, file_prefix: str):
        self.bucket_dir = bucket_dir
        self.file_prefix = file_prefix
        self.shard_index = 0
        self.rows_in_current_shard = 0
        self.current_handle = None

    def _open_next_shard(self) -> None:
        self.bucket_dir.mkdir(parents=True, exist_ok=True)
        self.shard_index += 1
        shard_name = f"{self.file_prefix}_shard_{self.shard_index:05d}.jsonl"
        shard_path = self.bucket_dir / shard_name
        self.current_handle = open(shard_path, "w", encoding="utf-8")
        self.rows_in_current_shard = 0

    def write(self, row: dict[str, Any]) -> None:
        if self.current_handle is None or self.rows_in_current_shard >= MAX_ROWS_PER_SHARD:
            self.close()
            self._open_next_shard()
        self.current_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.rows_in_current_shard += 1

    def close(self) -> None:
        if self.current_handle is not None:
            self.current_handle.close()
            self.current_handle = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize completion JSONL rows into BBQ conversation format, "
            "bucket by token count, and shard outputs."
        )
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help=(
            "Path to the input JSONL file with model responses. "
            "If missing and ending with '_results.jsonl', the script will try merging "
            "matching '*_results_<start>_<end>.jsonl' shards and "
            "'*_results_checkpoints_<start>_<end>' checkpoint folders from the same directory."
        ),
    )
    parser.add_argument(
        "--tokenizer_dir",
        default=DEFAULT_TOKENIZER_DIR,
        help="Tokenizer directory used for BBQ chat template token counting.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to the directory containing --input_file.",
    )
    return parser.parse_args()


def load_tokenizer(tokenizer_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    process_entry_module.TOKENIZER = tokenizer
    return tokenizer


def discover_result_range_sources(input_file: str) -> list[ResultRangeSource]:
    if not input_file.endswith("_results.jsonl"):
        raise FileNotFoundError(
            f"Input file not found: {input_file}. "
            "Automatic shard merge is only supported for missing '*_results.jsonl' targets."
        )

    input_dir = os.path.dirname(input_file) or "."
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    input_basename = os.path.basename(input_file)
    expected_prefix = input_basename[: -len("_results.jsonl")]

    shard_sources_by_range: dict[tuple[int, int], ResultRangeSource] = {}
    checkpoint_sources_by_range: dict[tuple[int, int], ResultRangeSource] = {}
    for file_name in os.listdir(input_dir):
        match = SHARDED_RESULTS_FILENAME_RE.fullmatch(file_name)
        if match is not None and match.group("prefix") == expected_prefix:
            start_idx = int(match.group("start"))
            end_idx = int(match.group("end"))
            shard_sources_by_range[(start_idx, end_idx)] = ResultRangeSource(
                start_idx=start_idx,
                end_idx=end_idx,
                path=os.path.join(input_dir, file_name),
                source_type="shard",
            )
            continue

        checkpoint_match = CHECKPOINT_DIRNAME_RE.fullmatch(file_name)
        if checkpoint_match is None or checkpoint_match.group("prefix") != expected_prefix:
            continue

        checkpoint_path = os.path.join(input_dir, file_name)
        if not os.path.isdir(checkpoint_path):
            continue

        start_idx = int(checkpoint_match.group("start"))
        end_idx = int(checkpoint_match.group("end"))
        checkpoint_sources_by_range[(start_idx, end_idx)] = ResultRangeSource(
            start_idx=start_idx,
            end_idx=end_idx,
            path=checkpoint_path,
            source_type="checkpoint",
        )

    if not shard_sources_by_range and not checkpoint_sources_by_range:
        raise FileNotFoundError(
            f"Input file not found: {input_file}. "
            f"No matching shards or checkpoint folders found in {input_dir} "
            f"for prefix '{expected_prefix}_results_<start>_<end>.jsonl'."
        )

    range_sources = list(shard_sources_by_range.values())
    for range_key, checkpoint_source in checkpoint_sources_by_range.items():
        if range_key not in shard_sources_by_range:
            range_sources.append(checkpoint_source)

    range_sources.sort(key=lambda source: (source.start_idx, source.end_idx))

    for i, source in enumerate(range_sources):
        start_idx = source.start_idx
        end_idx = source.end_idx
        if start_idx >= end_idx:
            raise ValueError(
                f"Invalid {source.source_type} range in {source.path}: "
                f"start ({start_idx}) must be smaller than end ({end_idx})."
            )

        if i == 0:
            continue

        prev_source = range_sources[i - 1]
        if start_idx < prev_source.end_idx:
            raise ValueError(
                "Overlapping result ranges detected: "
                f"{prev_source.path} ({prev_source.start_idx}, {prev_source.end_idx}) "
                f"and {source.path} ({start_idx}, {end_idx})."
            )

    return range_sources


def write_checkpoint_dir_to_jsonl(checkpoint_dir: str, output_handle) -> tuple[int, int]:
    line_count = 0
    skipped_file_count = 0
    checkpoint_files: list[tuple[int, str]] = []

    for file_name in os.listdir(checkpoint_dir):
        match = CHECKPOINT_FILENAME_RE.fullmatch(file_name)
        if match is None:
            continue
        checkpoint_files.append((int(match.group("index")), os.path.join(checkpoint_dir, file_name)))

    checkpoint_files.sort(key=lambda x: x[0])

    for _, checkpoint_path in checkpoint_files:
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f_in:
                checkpoint_item = json.load(f_in)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: Failed to load checkpoint {checkpoint_path}: {exc}. Skipping.")
            skipped_file_count += 1
            continue

        json.dump(checkpoint_item, output_handle, ensure_ascii=False)
        output_handle.write("\n")
        line_count += 1

    return line_count, skipped_file_count


def get_checkpoint_recovered_shard_path(checkpoint_dir: str) -> str:
    checkpoint_dirname = os.path.basename(os.path.normpath(checkpoint_dir))
    checkpoint_match = CHECKPOINT_DIRNAME_RE.fullmatch(checkpoint_dirname)
    if checkpoint_match is None:
        raise ValueError(f"Checkpoint directory name does not match expected pattern: {checkpoint_dir}")

    shard_filename = (
        f"{checkpoint_match.group('prefix')}_results_"
        f"{checkpoint_match.group('start')}_{checkpoint_match.group('end')}.jsonl"
    )
    return os.path.join(os.path.dirname(checkpoint_dir), shard_filename)


def compact_checkpoint_dir_to_shard(checkpoint_dir: str) -> tuple[str, int, int, bool]:
    shard_output_path = get_checkpoint_recovered_shard_path(checkpoint_dir)
    output_dir = os.path.dirname(shard_output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(shard_output_path):
        return shard_output_path, 0, 0, False

    file_descriptor, temp_output_path = tempfile.mkstemp(
        dir=output_dir,
        prefix=f".{os.path.basename(shard_output_path)}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as f_out:
            line_count, skipped_file_count = write_checkpoint_dir_to_jsonl(checkpoint_dir, f_out)
        os.replace(temp_output_path, shard_output_path)
    except Exception:
        try:
            os.unlink(temp_output_path)
        except FileNotFoundError:
            pass
        raise

    return shard_output_path, line_count, skipped_file_count, True


def merge_shard_file_to_output(shard_path: str, output_handle) -> int:
    line_count = 0
    with open(shard_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            if not line.strip():
                continue
            output_handle.write(line if line.endswith("\n") else f"{line}\n")
            line_count += 1
    return line_count


def merge_result_sources(range_sources: list[ResultRangeSource], merged_output_file: str) -> MergeRecoveryStats:
    output_dir = os.path.dirname(merged_output_file) or "."
    os.makedirs(output_dir, exist_ok=True)

    stats = MergeRecoveryStats()
    with open(merged_output_file, "w", encoding="utf-8") as f_out:
        prev_end_idx: Optional[int] = None
        for source in range_sources:
            if prev_end_idx is not None and source.start_idx > prev_end_idx:
                gap_range = f"[{prev_end_idx}, {source.start_idx})"
                stats.gap_ranges.append(gap_range)
                print(
                    "Warning: Unrecoverable gap in shard ranges detected: "
                    f"no shard or checkpoint data found for {gap_range}."
                )

            if source.source_type == "shard":
                stats.line_count += merge_shard_file_to_output(source.path, f_out)
                stats.existing_shard_count += 1
            elif source.source_type == "checkpoint":
                recovered_shard_path, recovered_lines, skipped_files, was_compacted = compact_checkpoint_dir_to_shard(
                    source.path
                )
                stats.skipped_checkpoint_file_count += skipped_files
                if was_compacted:
                    stats.checkpoint_dir_count += 1
                    stats.recovered_ranges.append(f"[{source.start_idx}, {source.end_idx})")
                    print(
                        f"Compacted checkpoint dir {source.path} into recovered shard "
                        f"{recovered_shard_path} with {recovered_lines} row(s)."
                    )
                stats.line_count += merge_shard_file_to_output(recovered_shard_path, f_out)
                stats.recovered_shard_count += 1
            else:
                raise ValueError(f"Unsupported merge source type: {source.source_type}")

            prev_end_idx = source.end_idx

    return stats


def resolve_or_merge_input_file(input_file: str) -> str:
    if os.path.exists(input_file):
        print(f"Using existing input file: {input_file}")
        return input_file

    print(f"Input file not found: {input_file}. Trying shard/checkpoint recovery fallback...")
    range_sources = discover_result_range_sources(input_file)
    start_idx = range_sources[0].start_idx
    end_idx = range_sources[-1].end_idx
    merge_stats = merge_result_sources(range_sources, input_file)

    summary_parts = [
        f"Merged {merge_stats.existing_shard_count} existing shard file(s)",
        f"compacted {merge_stats.checkpoint_dir_count} checkpoint folder(s)",
        f"merged {merge_stats.recovered_shard_count} recovered shard file(s)",
        f"into {input_file}",
        f"covered range: [{start_idx}, {end_idx})",
        f"non-empty rows written: {merge_stats.line_count}",
    ]
    print(". ".join(summary_parts) + ".")

    if merge_stats.recovered_ranges:
        print("Recovered checkpoint-backed ranges: " + ", ".join(merge_stats.recovered_ranges) + ".")

    if merge_stats.gap_ranges:
        print("Unrecoverable gaps skipped: " + ", ".join(merge_stats.gap_ranges) + ".")

    if merge_stats.skipped_checkpoint_file_count:
        print(f"Skipped unreadable checkpoint files: {merge_stats.skipped_checkpoint_file_count}.")

    return input_file


def parse_tools_from_system_prompt(system_content: str) -> tuple[str, list[Any]]:
    blocks = [match.group(1).strip() for match in TOOL_BLOCK_RE.finditer(system_content or "")]
    tool_block = max((block for block in blocks if block), key=len, default="")
    tools = []

    if tool_block:
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(tool_block):
            while idx < len(tool_block) and tool_block[idx].isspace():
                idx += 1
            if idx >= len(tool_block):
                break
            obj, idx = decoder.raw_decode(tool_block, idx)
            tools.append(obj)

    cleaned = TOOL_BLOCK_RE.sub("", system_content or "")
    for line in SYSTEM_BOILERPLATE_LINES:
        cleaned = cleaned.replace(line, "")
    cleaned = cleaned.strip()
    return cleaned, tools


def normalize_messages(messages: Any, stats: RunStats) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    if not isinstance(messages, list):
        return None, "messages is not a list"
    if not messages:
        return None, "messages is empty"

    conversation: list[dict[str, Any]] = []
    i = 0

    first_message = messages[0]
    if not isinstance(first_message, dict):
        return None, "first message is not an object"

    if first_message.get("role") == "system":
        system_turn: dict[str, Any] = {"role": "system"}
        try:
            system_content, tools = parse_tools_from_system_prompt(str(first_message.get("content", "")))
        except json.JSONDecodeError as exc:
            return None, f"failed to parse tools from system prompt: {exc}"
        if system_content:
            system_turn["content"] = system_content
        if tools:
            system_turn["tools"] = tools
        conversation.append(system_turn)
        i = 1

    while i < len(messages):
        message = messages[i]
        if not isinstance(message, dict):
            return None, f"message at index {i} is not an object"

        role = message.get("role")
        if role == "user":
            conversation.append({"role": "user", "content": str(message.get("content", ""))})
            i += 1
            continue

        if role == "assistant":
            think_parts: list[str] = []
            content_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []

            while i < len(messages):
                assistant_message = messages[i]
                if not isinstance(assistant_message, dict) or assistant_message.get("role") != "assistant":
                    break

                reasoning = assistant_message.get("reasoning_content")
                if reasoning:
                    think_parts.append(str(reasoning))

                content = assistant_message.get("content")
                if content:
                    content_parts.append(str(content))

                function_call = assistant_message.get("function_call")
                if function_call:
                    if not isinstance(function_call, dict):
                        return None, f"assistant function_call at index {i} is not an object"
                    tool_call: dict[str, Any] = {
                        "name": function_call.get("name"),
                        "arguments": function_call.get("arguments"),
                    }
                    call_id = function_call.get("call_id")
                    if call_id:
                        tool_call["id"] = call_id
                    tool_calls.append(tool_call)

                i += 1

            assistant_turn: dict[str, Any] = {"role": "assistant"}
            if think_parts:
                assistant_turn["think"] = "\n\n".join(think_parts)
            if content_parts:
                assistant_turn["content"] = "\n\n".join(content_parts)
            else:
                assistant_turn["content"] = ""
            if tool_calls:
                assistant_turn["tool_calls"] = tool_calls
            conversation.append(assistant_turn)
            continue

        if role in ("function", "tool"):
            tool_turn = {
                "role": "tool",
                "content": str(message.get("content", "")),
            }
            name = message.get("name")
            if name:
                tool_turn["name"] = name
            tool_call_id = message.get("tool_call_id")
            if tool_call_id:
                tool_turn["tool_call_id"] = tool_call_id
            conversation.append(tool_turn)
            i += 1
            continue

        stats.skipped_unknown_roles += 1
        i += 1

    while conversation and conversation[-1].get("role") == "user":
        conversation.pop()

    if not conversation:
        return None, "conversation is empty after normalization"
    return conversation, None


def bucket_for_token_count(token_count: int) -> str:
    if token_count <= 8_192:
        return "8k"
    if token_count <= 32_768:
        return "32k"
    if token_count <= 131_072:
        return "128k"
    if token_count <= 524_288:
        return "512k"
    return "above_512k"


def format_output_row(processed_entry: dict[str, Any], original_row: dict[str, Any]) -> dict[str, Any]:
    output_row = {
        "conversation": processed_entry["conversation"],
        "token_count_answer": processed_entry["token_count_answer"],
        "token_count_think": processed_entry["token_count_think"],
        "token_count": processed_entry["token_count"],
    }
    if "metadata" in original_row:
        output_row["metadata"] = original_row["metadata"]
    return output_row


def process_file(input_file: Path, output_dir: Path) -> RunStats:
    stats = RunStats()
    writers: dict[str, ShardWriter] = {}
    file_prefix = input_file.stem

    try:
        with input_file.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue

                stats.total_rows += 1

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    stats.json_failures += 1
                    stats.add_failure(line_no, "json_decode", str(exc))
                    continue

                conversation, normalization_error = normalize_messages(row.get("messages"), stats)
                if normalization_error is not None:
                    stats.normalization_failures += 1
                    stats.add_failure(line_no, "normalize", normalization_error)
                    continue

                normalized_entry = {"conversation": conversation}
                processed_entry, errors = process_entry_module.process_entry(
                    normalized_entry,
                    reasoning_effort="high",
                    identity_check=False,
                    compute_token_counts=True,
                )
                if errors:
                    stats.validation_failures += 1
                    stats.add_failure(line_no, "validate", "; ".join(errors))
                    continue

                output_row = format_output_row(processed_entry, row)
                bucket = bucket_for_token_count(processed_entry["token_count"])
                writer = writers.get(bucket)
                if writer is None:
                    bucket_dir = output_dir / bucket
                    writer = ShardWriter(bucket_dir=bucket_dir, file_prefix=f"{file_prefix}_{bucket}")
                    writers[bucket] = writer

                writer.write(output_row)
                stats.rows_written += 1
                stats.bucket_counts[bucket] = stats.bucket_counts.get(bucket, 0) + 1
    finally:
        for writer in writers.values():
            writer.close()

    return stats


def print_summary(stats: RunStats, output_dir: Path) -> None:
    print(f"Output directory: {output_dir}")
    print(f"Total non-empty rows read: {stats.total_rows}")
    print(f"Rows written: {stats.rows_written}")
    print(f"JSON decode failures: {stats.json_failures}")
    print(f"Normalization failures: {stats.normalization_failures}")
    print(f"Validation failures: {stats.validation_failures}")
    print(f"Unknown-role messages skipped during normalization: {stats.skipped_unknown_roles}")
    print("Bucket counts:")
    for bucket in ("8k", "32k", "128k", "512k", "above_512k"):
        print(f"  {bucket}: {stats.bucket_counts.get(bucket, 0)}")
    if stats.failure_examples:
        print("Failure examples:")
        for example in stats.failure_examples:
            print(f"  line {example.line_no} [{example.stage}] {example.details}")


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file).expanduser().resolve()
    resolved_input_file = Path(resolve_or_merge_input_file(str(input_file)))

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else resolved_input_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {args.tokenizer_dir}")
    load_tokenizer(args.tokenizer_dir)

    stats = process_file(input_file=resolved_input_file, output_dir=output_dir)
    print_summary(stats=stats, output_dir=output_dir)


if __name__ == "__main__":
    main()
