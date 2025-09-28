from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


def build_joint_per_test_markdown(
    results_dirs: List[str | Path],
    *,
    models: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    table_title: Optional[str] = "Joint@K per test (best-of-seeds, summary row per model)",
) -> str:
    """
    Build a Markdown table where each row is 'task_id::test_name' and columns are models.
    Each cell shows 'X/K ✓|×', where X = #seeds (K=len(results_dirs)) with Joint success:
       - reference_pass[test_name] == True AND failure_fail[test_name] == True (in the same seed)
    The final row '**Overall Joint@K**' summarizes per model as 'Y/N (P%)', where:
       - N = number of rows (tests) in the table
       - Y = number of tests with Joint@K success (>=1 seed) for that model
       - P = 100*Y/N

    Expects files named '{task_id}_tests_{llm}.json' with:
       test_results: { "reference_pass": [(name, bool), ...], "failure_fail": [(name, bool), ...] }
    """
    seeds = [_load_single_seed_tests(d) for d in results_dirs]
    K = len(seeds)

    # Universe discovery
    all_tasks, all_models = set(), set()
    for seed in seeds:
        all_tasks.update(seed.keys())
        for tblock in seed.values():
            all_models.update(tblock.keys())

    task_list = [t for t in (tasks if tasks is not None else sorted(all_tasks)) if t in all_tasks]
    model_list = [m for m in (models if models is not None else sorted(all_models)) if m in all_models]

    # Collect joint test names (those that appear in both reference_pass and failure_fail for some model/seed)
    rows = []
    joint_test_keys: List[Tuple[str, str]] = []  # (task_id, test_name)

    for task_id in task_list:
        ref_names, fail_names = set(), set()
        for seed in seeds:
            for m in seed.get(task_id, {}):
                ref_names.update(seed[task_id][m].get("reference_pass", {}).keys())
                fail_names.update(seed[task_id][m].get("failure_fail", {}).keys())
        names = sorted(ref_names & fail_names)
        for test_name in names:
            row = [f"{task_id}::{test_name}"]
            for m in model_list:
                x = 0
                for seed in seeds:
                    ok_ref = seed.get(task_id, {}).get(m, {}).get("reference_pass", {}).get(test_name, False)
                    ok_fail = seed.get(task_id, {}).get(m, {}).get("failure_fail", {}).get(test_name, False)
                    if ok_ref and ok_fail:
                        x += 1
                row.append(f"{x}/{K} {'✓' if x >= 1 else '×'}")
            rows.append(row)
            joint_test_keys.append((task_id, test_name))

    headers = ["Task::Test"] + model_list
    df = pd.DataFrame(rows, columns=headers)

    # Build summary row per model
    if joint_test_keys:
        N = len(joint_test_keys)
        summary = ["**Overall Joint@{}**".format(K)]
        for j, m in enumerate(model_list, start=1):
            # Count tests where this model had success in >=1 seed
            Y = 0
            for _, _ in joint_test_keys:
                # Row index independent of order: search df row by label
                # Faster to recompute directly from 'rows' would require tracking; here we parse cell.
                # We'll compute from df cells to keep it simple.
                pass
        # Compute Y from df values:
        for m in model_list:
            col = df[m] if not df.empty else []
            Y = 0
            for cell in col:
                # cell format: "x/K ✓" or "x/K ×"
                x_str = str(cell).split("/", 1)[0].strip()
                try:
                    x_val = int(x_str)
                except Exception:
                    x_val = 0
                if x_val >= 1:
                    Y += 1
            pct = (100.0 * Y / N) if N else 0.0
            summary.append(f"{Y}/{N} ({pct:.1f}%)")
        df.loc[len(df)] = summary  # append summary row

    title = f"### {table_title} (K={K})\n\n"
    return title + df.to_markdown(index=False) + "\n"


# ---------------------- internals ----------------------

def _load_single_seed_tests(results_dir: str | Path) -> Dict[str, Dict[str, Dict[str, Dict[str, bool]]]]:
    """
    For one seed results directory, load '{task}_tests_{llm}.json' and return:
        seed[task][model]['reference_pass'][test_name] -> bool
        seed[task][model]['failure_fail'][test_name]   -> bool
    Missing sections are treated as empty.
    """
    out: Dict[str, Dict[str, Dict[str, Dict[str, bool]]]] = {}
    d = Path(results_dir)
    for p in d.glob("*_tests_*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        task_id, model_name = _parse_tests_filename(p.name)
        tr = data.get("test_results", {}) or {}
        ref_pairs = tr.get("reference_pass", []) or []
        fail_pairs = tr.get("failure_fail", []) or []
        ref_map = {name: bool(ok) for name, ok in ref_pairs}
        fail_map = {name: bool(ok) for name, ok in fail_pairs}
        out.setdefault(task_id, {}).setdefault(model_name, {})["reference_pass"] = ref_map
        out.setdefault(task_id, {}).setdefault(model_name, {})["failure_fail"] = fail_map
    return out


def _parse_tests_filename(filename: str) -> Tuple[str, str]:
    # "{task_id}_tests_{llm}.json" -> (task_id, llm)
    stem = filename[:-5] if filename.endswith(".json") else filename
    left, model = stem.split("_tests_", 1)
    return left, model
