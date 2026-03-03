import argparse
import importlib.util
import itertools
import json
import os
import time
from typing import Any, Iterator

"""
Hero-run prompt generator restricted to selected AI occupations.

Sampling strategy (coverage-first):
  Phase 1 — Greedy set-cover: pick task triples that cover the most uncovered
    tasks first, until every task with a matched server appears in at least one
    prompt.  Uses at most ceil(N / num_tasks) prompts per occupation.
  Phase 2 — Fill to budget: add further triples (in sorted index order) until
    max_per_occupation is reached or all combinations are exhausted.

Server selection: for each task in a triple the matched server with the highest
similarity_score is chosen (ties broken by server_id lexicographically).  This
yields exactly ONE prompt per task-triple, with no combinatorial explosion.

Example:
  python step1.1_hero_selected.py --no_refs --job_name hero_run_v1
"""

# ---------------------------------------------------------------------------
# Import reusable components from step1.1_gen_questions_from_onet_tasks.py
# (importlib needed because the filename contains dots)
# ---------------------------------------------------------------------------

_STEP1_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "step1.1_gen_questions_from_onet_tasks.py",
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
        default=30,
        help="Maximum prompts per occupation.",
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
        "--withheld",
        action="store_true",
        help="Use withheld-info variant of the generation prompt (genq_from_onet_tasks_withheld.md).",
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
# Coverage-first sampling
# ---------------------------------------------------------------------------


def coverage_first_sample(
    tasks: list[dict[str, Any]],
    num_tasks: int,
    max_per_occ: int,
) -> list[tuple[int, ...]]:
    """Return up to max_per_occ task-index tuples for one occupation.

    Phase 1: greedy set-cover — pick combos that cover the most uncovered task
    indices first (ties broken by earliest index order).
    Phase 2: fill remaining budget in sorted index order.
    """
    n = len(tasks)
    if n < num_tasks or max_per_occ <= 0:
        return []

    all_combos = list(itertools.combinations(range(n), num_tasks))

    # Phase 1 — greedy coverage
    uncovered: set[int] = set(range(n))
    selected: list[tuple[int, ...]] = []
    selected_set: set[tuple[int, ...]] = set()
    # Work on a copy so we can remove without affecting all_combos
    remaining = list(all_combos)

    while uncovered and remaining:
        best = max(remaining, key=lambda c: len(set(c) & uncovered))
        new_coverage = set(best) & uncovered
        if not new_coverage:
            break
        selected.append(best)
        selected_set.add(best)
        uncovered -= new_coverage
        remaining.remove(best)

        if len(selected) >= max_per_occ:
            break

    # Phase 2 — fill to budget in deterministic sorted order
    if len(selected) < max_per_occ:
        for combo in all_combos:
            if len(selected) >= max_per_occ:
                break
            if combo not in selected_set:
                selected.append(combo)
                selected_set.add(combo)

    return selected[:max_per_occ]


def select_server_id(
    task: dict[str, Any],
    combo_rank: int,
    avoid: set[str] | None = None,
) -> str | None:
    """Pick a server for a task, cycling by combo_rank to diversify across prompts.

    Servers are sorted by descending similarity_score (ties broken by server_id).
    If `avoid` is given, prefer a server not in that set; fall back to cycling
    through all servers when no alternative exists.
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
        if alternatives:
            return alternatives[combo_rank % len(alternatives)]["server_id"]
    return servers_sorted[combo_rank % len(servers_sorted)]["server_id"]


# ---------------------------------------------------------------------------
# Hero combo iterator
# ---------------------------------------------------------------------------


def iter_hero_combos(
    occupation_to_tasks: dict[str, list[dict[str, Any]]],
    valid_onet_codes: list[str],
    num_tasks: int,
    max_per_occ: int,
) -> Iterator[dict[str, Any]]:
    """Yield combo records using coverage-first sampling with diversified server selection."""
    for onet_code in sorted(valid_onet_codes):
        tasks = sorted(occupation_to_tasks[onet_code], key=lambda t: t.get("task_id", ""))
        selected_combos = coverage_first_sample(tasks, num_tasks, max_per_occ)

        for combo_rank, combo_indices in enumerate(selected_combos):
            selected_tasks = [tasks[i] for i in combo_indices]
            server_ids = []
            skip = False
            used_in_combo: set[str] = set()
            for task in selected_tasks:
                sid = select_server_id(task, combo_rank, avoid=used_in_combo)
                if sid is None:
                    skip = True
                    break
                server_ids.append(sid)
                used_in_combo.add(sid)
            if skip:
                continue

            yield {
                "onet_code": onet_code,
                "task_indices": combo_indices,
                "tasks": selected_tasks,
                "selected_server_ids": tuple(server_ids),
            }


def count_hero_combos(
    occupation_to_tasks: dict[str, list[dict[str, Any]]],
    valid_onet_codes: list[str],
    num_tasks: int,
    max_per_occ: int,
) -> int:
    """Count total prompts and print per-occupation breakdown."""
    per_occ: dict[str, int] = {}
    for onet_code in sorted(valid_onet_codes):
        tasks = sorted(
            occupation_to_tasks[onet_code], key=lambda t: t.get("task_id", "")
        )
        selected_combos = coverage_first_sample(tasks, num_tasks, max_per_occ)
        # Subtract any combos where a task has no loadable server
        valid_count = 0
        for combo_indices in selected_combos:
            selected_tasks = [tasks[i] for i in combo_indices]
            if all(select_server_id(t, combo_rank=0) is not None for t in selected_tasks):
                valid_count += 1
        per_occ[onet_code] = valid_count

    total = sum(per_occ.values())
    print(f"\nPer-occupation prompt counts (num_tasks={num_tasks}, max={max_per_occ}):")
    for code, cnt in sorted(per_occ.items()):
        print(f"  {code}: {cnt}")
    print(f"Total prompts: {total}\n")
    return total


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------


def prepare_hero_output_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    output_dirname = (
        f"hero_selected_{args.num_tasks}tasks_{args.max_per_occupation}max_{args.timestamp}"
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

    # --withheld selects the withheld-info prompt variant; --no_refs still
    # controls whether task references are loaded and inserted at runtime.
    paths = resolve_paths(script_dir, no_refs=False, withheld=args.withheld)
    inputs = load_inputs(paths, args)

    # Filter tasks to those with loadable servers, then build occupation index
    tasks_filtered = filter_tasks_with_loadable_servers(inputs.tasks_list, inputs.server_index)
    occupation_to_tasks = create_occupation_to_tasks_index(tasks_filtered)

    # Keep only selected occupations that have enough tasks
    valid_onet_codes = sorted(
        code
        for code, tasks in occupation_to_tasks.items()
        if code in selected_onet_codes and len(tasks) >= args.num_tasks
    )

    missing = selected_onet_codes - set(valid_onet_codes)
    if missing:
        print(
            f"Warning: {len(missing)} selected occupation(s) have fewer than "
            f"{args.num_tasks} loadable tasks and will be skipped: {sorted(missing)}"
        )
    print(
        f"Generating prompts for {len(valid_onet_codes)} occupations "
        f"(max {args.max_per_occupation} per occupation)"
    )

    total_prompts = count_hero_combos(
        occupation_to_tasks, valid_onet_codes, args.num_tasks, args.max_per_occupation
    )

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
            occupation_to_tasks, valid_onet_codes, args.num_tasks, args.max_per_occupation
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
