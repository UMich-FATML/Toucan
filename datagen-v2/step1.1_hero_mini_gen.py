import argparse
import importlib.util
import itertools
import json
import math
import os
import time
from collections import defaultdict
from typing import Any, Iterator

"""
Hero-run prompt generator for selected AI occupations.

Sampling strategy:
  1) Compute number of possible task-combo scenarios per selected occupation:
       C(num_tasks_in_occupation, num_tasks)
  2) Keep top-K occupations by scenario count.
  3) Per occupation, sample up to max_per_occupation scenarios with a
     coverage+diversity heuristic over task combinations.

Server selection: per sampled task-triple, servers are selected from matched
options while favoring lower overall usage (ties follow similarity_score then
server_id). This increases server diversity while yielding exactly one prompt
per sampled scenario.

Example:
  python step1.1_hero_mini_gen.py --no_refs --job_name hero_run_v1
"""

# ---------------------------------------------------------------------------
# Import reusable components from step1.1_gen_questions_from_onet_tasks.py
# (importlib needed because the filename contains dots)
# ---------------------------------------------------------------------------

_STEP1_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "step1.1_gen_prompts_from_onet_tasks.py",
)
_spec = importlib.util.spec_from_file_location("_step1_1_mod", _STEP1_PATH)
_step1_1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_step1_1)

# Aliases
load_inputs = _step1_1.load_inputs
filter_tasks_with_loadable_servers = _step1_1.filter_tasks_with_loadable_servers
create_occupation_to_tasks_index = _step1_1.create_occupation_to_tasks_index
build_prompt_record = _step1_1.build_prompt_record
generate_and_write_prompts_streaming = _step1_1.generate_and_write_prompts_streaming
TaskRefLookupStats = _step1_1.TaskRefLookupStats
load_task_refs_index = _step1_1.load_task_refs_index
resolve_paths = _step1_1.resolve_paths
save_generation_args = _step1_1.save_generation_args
init_generation_settings = _step1_1.init_generation_settings


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hero-run prompt generator for selected AI occupations."
    )
    parser.add_argument("--num_tasks", type=int, default=3, help="Tasks per prompt.")
    parser.add_argument(
        "--num_tools",
        type=int,
        default=None,
        help="Minimum tools required in model output; defaults to --num_tasks.",
    )
    parser.add_argument(
        "--max_per_occupation",
        type=int,
        default=100,
        help="Maximum sampled scenarios per selected occupation.",
    )
    parser.add_argument(
        "--top_occupations",
        type=int,
        default=9,
        help="Number of occupations to keep, ranked by possible scenario count.",
    )
    parser.add_argument(
        "--selected_occupations_file",
        type=str,
        default=None,
        help="Path to JSON list of {occupation_code, occupation_title}. "
        "Defaults to ../data/selected_ai_occupations.json relative to this script.",
    )
    parser.add_argument("--output_folder", type=str, default="../data")
    parser.add_argument("--job_name", type=str, default=None)
    parser.add_argument("--timestamp", type=int, default=int(time.time()))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--mcp_servers_dir",
        type=str,
        default="../mcp_servers/smithery_mcp_servers_0210",
    )
    parser.add_argument("--no_refs", action="store_true")
    parser.add_argument(
        "--self_contained",
        action="store_true",
        help="Use self contained variant of the generation prompt (gen_self_contained_q_from_onet_tasks_md).",
    )
    parser.add_argument(
        "--count_only",
        action="store_true",
        help="Only print selected occupations and total prompt count; do not generate JSONL output.",
    )

    args = parser.parse_args()
    if args.num_tools is None:
        args.num_tools = args.num_tasks
    # Needed for compatibility with step1.1 helpers (e.g. prepare_output_paths).
    args.total_prompts = None
    return args


# ---------------------------------------------------------------------------
# Selected-occupation loading
# ---------------------------------------------------------------------------


def load_selected_onet_codes(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    codes = {entry["occupation_code"] for entry in data}
    print(f"Loaded {len(codes)} selected occupation codes from {path}")
    return codes


# ---------------------------------------------------------------------------
# Top-K occupation selection by scenario count
# ---------------------------------------------------------------------------


def scenario_count(task_count: int, num_tasks: int) -> int:
    if task_count < num_tasks:
        return 0
    return math.comb(task_count, num_tasks)


def select_top_occupations_by_scenarios(
    occupation_to_tasks: dict[str, list[dict[str, Any]]],
    selected_onet_codes: set[str],
    num_tasks: int,
    top_occupations: int,
) -> tuple[list[str], dict[str, int], dict[str, int], set[str]]:
    """Return top occupations by task-combo count among selected occupation codes."""
    candidates: list[tuple[str, int, int]] = []
    dropped_insufficient_tasks: set[str] = set()

    for code in sorted(selected_onet_codes):
        tasks = occupation_to_tasks.get(code, [])
        task_count = len(tasks)
        combos = scenario_count(task_count, num_tasks)
        if combos <= 0:
            dropped_insufficient_tasks.add(code)
            continue
        candidates.append((code, combos, task_count))

    ranked = sorted(candidates, key=lambda item: (-item[1], item[0]))
    selected_ranked = ranked[:top_occupations]

    selected_codes = [code for code, _, _ in selected_ranked]
    combos_by_code = {code: combos for code, combos, _ in selected_ranked}
    task_count_by_code = {code: task_count for code, _, task_count in selected_ranked}
    return selected_codes, combos_by_code, task_count_by_code, dropped_insufficient_tasks


def select_server_id(
    task: dict[str, Any],
    combo_rank: int,
    avoid: set[str] | None = None,
    usage_by_server: dict[str, int] | None = None,
) -> str | None:
    """Pick a server for a task, preferring less-used servers for diversity.

    Servers are sorted by descending similarity_score (ties broken by server_id).
    If `avoid` is given, prefer a server not in that set; fall back to cycling
    through all servers when no alternative exists. If `usage_by_server` is
    provided, choose from least-used server IDs first.
    """
    servers = [s for s in task.get("matched_servers", []) if s.get("server_id")]
    if not servers:
        return None
    servers_sorted = sorted(
        servers,
        key=lambda s: (-s.get("similarity_score", 0.0), s.get("server_id", "")),
    )
    if avoid:
        alternatives = [s for s in servers_sorted if s["server_id"] not in avoid]
        candidates = alternatives if alternatives else servers_sorted
    else:
        candidates = servers_sorted

    if usage_by_server is None:
        return candidates[combo_rank % len(candidates)]["server_id"]

    min_usage = min(usage_by_server.get(s["server_id"], 0) for s in candidates)
    least_used = [
        s for s in candidates if usage_by_server.get(s["server_id"], 0) == min_usage
    ]
    return least_used[combo_rank % len(least_used)]["server_id"]


def sample_diverse_task_combos(
    tasks: list[dict[str, Any]],
    num_tasks: int,
    max_per_occ: int,
) -> list[tuple[int, ...]]:
    """Sample task-index combinations with coverage-first then diversity-first selection."""
    n_tasks = len(tasks)
    if n_tasks < num_tasks or max_per_occ <= 0:
        return []

    all_combos = list(itertools.combinations(range(n_tasks), num_tasks))
    target = min(max_per_occ, len(all_combos))
    if target == len(all_combos):
        return all_combos

    task_rarity: dict[int, float] = {}
    for idx, task in enumerate(tasks):
        unique_server_count = len(
            {
                s.get("server_id")
                for s in task.get("matched_servers", [])
                if s.get("server_id")
            }
        )
        task_rarity[idx] = 1.0 / max(1, unique_server_count)

    combo_meta: list[dict[str, Any]] = []
    for combo in all_combos:
        pairs = tuple(itertools.combinations(combo, 2))
        combo_meta.append(
            {
                "combo": combo,
                "set": set(combo),
                "pairs": pairs,
                "rarity": sum(task_rarity[i] for i in combo),
                "tie": tuple(-i for i in combo),
            }
        )

    selected: list[tuple[int, ...]] = []
    selected_set: set[tuple[int, ...]] = set()
    selected_combo_sets: list[set[int]] = []
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    uncovered: set[int] = set(range(n_tasks))

    def register(meta: dict[str, Any]) -> None:
        combo = meta["combo"]
        selected.append(combo)
        selected_set.add(combo)
        selected_combo_sets.append(meta["set"])
        uncovered.difference_update(meta["set"])
        for pair in meta["pairs"]:
            pair_counts[pair] += 1

    # Phase 1: ensure task coverage across sampled combos.
    while len(selected) < target and uncovered:
        best_meta = None
        best_score = None
        for meta in combo_meta:
            combo = meta["combo"]
            if combo in selected_set:
                continue

            cover = len(meta["set"] & uncovered)
            if cover == 0:
                continue

            unseen_pairs = sum(1 for pair in meta["pairs"] if pair_counts[pair] == 0)
            pair_penalty = sum(pair_counts[pair] for pair in meta["pairs"])
            score = (cover, unseen_pairs, meta["rarity"], -pair_penalty, meta["tie"])

            if best_score is None or score > best_score:
                best_score = score
                best_meta = meta

        if best_meta is None:
            break
        register(best_meta)

    # Phase 2: maximize novelty against existing sampled combos.
    while len(selected) < target:
        best_meta = None
        best_score = None

        for meta in combo_meta:
            combo = meta["combo"]
            if combo in selected_set:
                continue

            unseen_pairs = sum(1 for pair in meta["pairs"] if pair_counts[pair] == 0)
            pair_penalty = sum(pair_counts[pair] for pair in meta["pairs"])

            if selected_combo_sets:
                dists = [
                    num_tasks - len(meta["set"] & prev_set)
                    for prev_set in selected_combo_sets
                ]
                min_dist = min(dists)
                avg_dist = sum(dists) / len(dists)
            else:
                min_dist = num_tasks
                avg_dist = float(num_tasks)

            score = (
                min_dist,
                avg_dist,
                unseen_pairs,
                meta["rarity"],
                -pair_penalty,
                meta["tie"],
            )

            if best_score is None or score > best_score:
                best_score = score
                best_meta = meta

        if best_meta is None:
            break
        register(best_meta)

    return selected


def build_sampled_combo_records(
    occupation_to_tasks: dict[str, list[dict[str, Any]]],
    selected_onet_codes: list[str],
    num_tasks: int,
    max_per_occ: int,
    server_index: dict[str, dict[str, Any]] | None = None,
    min_tools: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build sampled combo records per occupation with server-diverse assignments.

    If ``server_index`` and ``min_tools`` are provided, combos whose selected
    servers collectively offer fewer than ``min_tools`` unique tools are skipped.
    This prevents generating prompts that ask the LLM for multi-tool scenarios
    when insufficient tools are actually available.
    """
    sampled_by_code: dict[str, list[dict[str, Any]]] = {}
    skipped_insufficient_tools = 0

    for onet_code in selected_onet_codes:
        tasks = sorted(occupation_to_tasks[onet_code], key=lambda t: t.get("task_id", ""))
        sampled_indices = sample_diverse_task_combos(tasks, num_tasks, max_per_occ)

        server_usage: dict[str, int] = defaultdict(int)
        records: list[dict[str, Any]] = []

        for combo_rank, combo_indices in enumerate(sampled_indices):
            selected_tasks = [tasks[i] for i in combo_indices]
            used_in_combo: set[str] = set()
            server_ids: list[str] = []
            skip = False

            for task in selected_tasks:
                sid = select_server_id(
                    task=task,
                    combo_rank=combo_rank,
                    avoid=used_in_combo,
                    usage_by_server=server_usage,
                )
                if sid is None:
                    skip = True
                    break
                server_ids.append(sid)
                used_in_combo.add(sid)

            if skip:
                continue

            # Check that the selected servers collectively have enough tools
            if server_index is not None and min_tools is not None:
                unique_tools: set[str] = set()
                for sid in set(server_ids):
                    server_data = server_index.get(sid, {})
                    for tool in server_data.get("server", {}).get("tools", []):
                        unique_tools.add(tool.get("name", ""))
                if len(unique_tools) < min_tools:
                    skipped_insufficient_tools += 1
                    continue

            for sid in server_ids:
                server_usage[sid] += 1

            records.append(
                {
                    "onet_code": onet_code,
                    "task_indices": combo_indices,
                    "tasks": selected_tasks,
                    "selected_server_ids": tuple(server_ids),
                }
            )

        sampled_by_code[onet_code] = records

    if skipped_insufficient_tools > 0:
        print(
            f"Warning: Skipped {skipped_insufficient_tools} combo(s) whose selected "
            f"servers had fewer than {min_tools} unique tools."
        )

    return sampled_by_code


# ---------------------------------------------------------------------------
# Hero combo iterator
# ---------------------------------------------------------------------------


def iter_hero_combos(
    sampled_combos_by_code: dict[str, list[dict[str, Any]]],
    selected_onet_codes: list[str],
) -> Iterator[dict[str, Any]]:
    """Yield pre-sampled combo records in occupation order."""
    for onet_code in selected_onet_codes:
        for record in sampled_combos_by_code.get(onet_code, []):
            yield record


def count_hero_combos(
    selected_onet_codes: list[str],
    sampled_combos_by_code: dict[str, list[dict[str, Any]]],
    combos_by_code: dict[str, int],
    task_count_by_code: dict[str, int],
    num_tasks: int,
    max_per_occ: int,
) -> int:
    """Count sampled prompts and print per-occupation prompt + server diversity."""
    total = 0
    global_unique_servers: set[str] = set()
    print(
        f"\nSelected occupations after diversity sampling "
        f"(num_tasks={num_tasks}, max_per_occupation={max_per_occ}):"
    )
    for code in selected_onet_codes:
        sampled_records = sampled_combos_by_code.get(code, [])
        sampled_count = len(sampled_records)
        total += sampled_count

        unique_servers = {
            sid
            for rec in sampled_records
            for sid in rec.get("selected_server_ids", ())
            if sid
        }
        unique_server_tuples = {
            rec.get("selected_server_ids", ())
            for rec in sampled_records
            if rec.get("selected_server_ids")
        }
        global_unique_servers.update(unique_servers)
        possible = combos_by_code.get(code, 0)

        print(
            f"  {code}: sampled={sampled_count}/{possible} prompts "
            f"({task_count_by_code.get(code, 0)} tasks), "
            f"server_diversity={len(unique_servers)} unique servers, "
            f"unique_server_assignments={len(unique_server_tuples)}"
        )
    print(f"Total prompts: {total}")
    print(
        "Total unique servers across all selected occupations: "
        f"{len(global_unique_servers)}\n"
    )
    return total


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------


def prepare_hero_output_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    output_dirname = (
        f"hero_top{args.top_occupations}_{args.num_tasks}tasks_"
        f"{args.max_per_occupation}max_diversity_{args.timestamp}"
    )
    output_base_dir = os.path.join(args.output_folder, args.job_name or output_dirname)
    os.makedirs(output_base_dir, exist_ok=True)
    output_file_path = os.path.join(output_base_dir, f"{output_dirname}_prepared.jsonl")
    args_file_path = os.path.join(output_base_dir, "generation_args.json")
    return output_base_dir, output_file_path, args_file_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    print("Hero Run: Selected-Occupation O*NET Prompt Generation")
    print(f"Arguments: {args}\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve selected-occupations file
    if args.selected_occupations_file is None:
        args.selected_occupations_file = os.path.normpath(
            os.path.join(script_dir, "..", "data", "selected_ai_occupations.json")
        )
    selected_onet_codes = load_selected_onet_codes(args.selected_occupations_file)

    ##--no_refs still
    # controls whether task references are loaded and inserted at runtime.
    paths = resolve_paths(script_dir, no_refs=args.no_refs, self_contained=args.self_contained)
    inputs = load_inputs(paths, args)

    # Filter tasks to those with loadable servers, then build occupation index
    tasks_filtered = filter_tasks_with_loadable_servers(inputs.tasks_list, inputs.server_index)
    occupation_to_tasks = create_occupation_to_tasks_index(tasks_filtered)

    # Keep top-K selected occupations by number of possible task combinations.
    selected_codes, combos_by_code, task_count_by_code, dropped_insufficient = (
        select_top_occupations_by_scenarios(
            occupation_to_tasks=occupation_to_tasks,
            selected_onet_codes=selected_onet_codes,
            num_tasks=args.num_tasks,
            top_occupations=args.top_occupations,
        )
    )

    if dropped_insufficient:
        print(
            f"Warning: {len(dropped_insufficient)} selected occupation(s) have fewer than "
            f"{args.num_tasks} loadable tasks and were skipped: {sorted(dropped_insufficient)}"
        )

    if not selected_codes:
        print("No occupations available after filtering; exiting.")
        return

    if len(selected_codes) < args.top_occupations:
        print(
            f"Warning: Requested top {args.top_occupations} occupations, "
            f"but only {len(selected_codes)} are available."
        )

    print(
        f"Generating prompts for top {len(selected_codes)} selected occupations "
        f"by scenario count (up to {args.max_per_occupation} sampled scenarios per occupation)."
    )

    sampled_combos_by_code = build_sampled_combo_records(
        occupation_to_tasks=occupation_to_tasks,
        selected_onet_codes=selected_codes,
        num_tasks=args.num_tasks,
        max_per_occ=args.max_per_occupation,
        server_index=inputs.server_index,
        min_tools=args.num_tools,
    )

    total_prompts = count_hero_combos(
        selected_codes,
        sampled_combos_by_code,
        combos_by_code,
        task_count_by_code,
        args.num_tasks,
        args.max_per_occupation,
    )

    if args.count_only:
        print("Count-only mode enabled; no output file generated.")
        return

    output_base_dir, output_file_path, args_file_path = prepare_hero_output_paths(args)
    args_dict = save_generation_args(args, args_file_path)

    task_refs_index = None
    ref_lookup_stats = TaskRefLookupStats()
    if not args.no_refs:
        print(f"Loading task refs from {paths['task_refs_path']}...")
        task_refs_index = load_task_refs_index(paths["task_refs_path"])

    worker_count = init_generation_settings(args)

    def build_row_fn(i: int, combo_record: dict[str, Any]) -> dict[str, Any]:
        return build_prompt_record(
            i=i,
            combo_record=combo_record,
            occupation_dict=inputs.occupation_dict,
            server_index=inputs.server_index,
            prompt_template=inputs.prompt_template,
            args=args,
            args_dict=args_dict,
            task_refs_index=task_refs_index,
            ref_lookup_stats=ref_lookup_stats,
        )

    written = generate_and_write_prompts_streaming(
        total_prompts=total_prompts,
        worker_count=worker_count,
        combo_iterator=iter_hero_combos(
            sampled_combos_by_code, selected_codes
        ),
        build_row_fn=build_row_fn,
        output_file_path=output_file_path,
    )

    print(f"\nDone. Wrote {written} prompts to:\n  {output_file_path}")
    if not args.no_refs:
        print(
            f"Task ref lookup: requested={ref_lookup_stats.requested}, "
            f"hits={ref_lookup_stats.hits}, "
            f"empty={ref_lookup_stats.empty_results}, "
            f"misses={ref_lookup_stats.misses}"
        )


if __name__ == "__main__":
    main()
