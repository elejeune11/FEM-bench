from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

def build_sorted_joint_test_table_with_colors(
    results_dirs: List[str | Path],
    *,
    models: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    table_title: str = "Joint@K per test (best of seeds) — ordered easiest→hardest",
) -> str:
    """
    Build a Markdown table for individual tests (rows = 'task_id::test_name', columns = models).
    Each cell shows 'X/K ✓|×', where X = number of seeds (K=len(results_dirs)) where the model
    achieved Joint success for that test:
        Joint success in a seed := reference_pass[test_name] == True AND failure_fail[test_name] == True

    Sorting (easiest→hardest) uses strong tiebreakers:
      1) #models with >=1 Joint success (desc)
      2) total Joint successes across all (models × seeds) (desc)
      3) overall Joint success rate across all (models × seeds) (desc)
      4) 'task_id::test_name' (asc)

    The left column (Task::Test) is color-coded by **coverage**:
        coverage = (# of models with >=1 Joint success) / (# of models)
        > 75% → green, > 50% → blue, > 0% → yellow, otherwise red.

    Expects files named '{task_id}_tests_{llm}.json' with:
        test_results: {
           "reference_pass": [(test_name, bool), ...],
           "failure_fail":   [(test_name, bool), ...]
        }
    """
    # ---- Load per-seed test results ----
    seed_maps = [_load_single_seed_tests(d) for d in results_dirs]
    K = len(seed_maps)

    # Universe discovery
    all_tasks, all_models = set(), set()
    for s in seed_maps:
        all_tasks.update(s.keys())
        for tblock in s.values():
            all_models.update(tblock.keys())

    task_list = [t for t in (tasks if tasks is not None else sorted(all_tasks)) if t in all_tasks]
    model_list = [m for m in (models if models is not None else sorted(all_models)) if m in all_models]

    # Build rows for every joint test (present in both ref & fail sets)
    rows: List[List[str]] = []
    sort_keys = []  # store tuples for sorting
    joint_rows_keys: List[Tuple[str, str]] = []  # (task_id, test_name)

    for task_id in task_list:
        # collect union of ref/fail test names across all models & seeds
        ref_names, fail_names = set(), set()
        for s in seed_maps:
            for m in s.get(task_id, {}):
                ref_names.update(s[task_id][m].get("reference_pass", {}).keys())
                fail_names.update(s[task_id][m].get("failure_fail", {}).keys())
        joint_names = sorted(ref_names & fail_names)
        if not joint_names:
            continue

        for test_name in joint_names:
            # Count Joint successes per model
            cell_strings = []
            models_with_any = 0
            total_joint_successes = 0

            for m in model_list:
                x = 0
                for s in seed_maps:
                    ok_ref = s.get(task_id, {}).get(m, {}).get("reference_pass", {}).get(test_name, False)
                    ok_fail = s.get(task_id, {}).get(m, {}).get("failure_fail", {}).get(test_name, False)
                    if ok_ref and ok_fail:
                        x += 1
                total_joint_successes += x
                any_ok = (x >= 1)
                if any_ok:
                    models_with_any += 1
                cell_strings.append(f"{x}/{K} {'✓' if any_ok else '×'}")

            # --- changed: color by coverage (models_with_any / #models) ---
            denom_models = max(len(model_list), 1)
            coverage = models_with_any / denom_models

            # we still compute joint_rate for the tertiary tiebreaker (unchanged)
            denom_all = max(len(model_list) * K, 1)
            joint_rate = total_joint_successes / denom_all

            # colored label by coverage
            label = f"{task_id}::{test_name}"
            label_colored = _color_label(label, coverage)

            # store sort key (negative for desc on numeric)
            sort_keys.append((
                -models_with_any,        # 1) coverage (primary)
                -total_joint_successes,  # 2) total successes
                -joint_rate,             # 3) success rate across models×seeds
                label                    # 4) stable tie-break
            ))
            rows.append([label_colored] + cell_strings)
            joint_rows_keys.append((task_id, test_name))

    # Sort rows easiest→hardest
    rows_sorted = [x for _, x in sorted(zip(sort_keys, rows), key=lambda p: p[0])]

    # Summary row per model: how many tests had >=1 Joint success (and percentage)
    summary = ["**Overall Joint@{}**".format(K)]
    N = len(rows_sorted)
    for j, m in enumerate(model_list, start=1):
        Y = 0
        for r in rows_sorted:
            cell = r[j]  # like "2/K ✓" or "0/K ×"
            x_str = str(cell).split("/", 1)[0].strip() if "/" in str(cell) else "0"
            try:
                x_val = int(x_str)
            except Exception:
                x_val = 0
            if x_val >= 1:
                Y += 1
        pct = (100.0 * Y / N) if N else 0.0
        summary.append(f"{Y}/{N} ({pct:.1f}%)")
    headers = ["Task::Test"] + model_list
    df = pd.DataFrame(rows_sorted, columns=headers)
    df.loc[len(df)] = summary

    title = f"### {table_title}\n\n"
    return title + df.to_markdown(index=False) + "\n"


# -------------------- internals --------------------

def _load_single_seed_tests(results_dir: str | Path) -> Dict[str, Dict[str, Dict[str, Dict[str, bool]]]]:
    """
    Load one seed dir of '{task}_tests_{llm}.json' into:
       seed[task][model]['reference_pass'][test_name] -> bool
       seed[task][model]['failure_fail'][test_name]   -> bool
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
    stem = filename[:-5] if filename.endswith(".json") else filename
    left, model = stem.split("_tests_", 1)
    return left, model


def _color_label(text: str, coverage: float) -> str:
    """
    Emoji code for coverage:
      100% → 🟩
      >0% but <100% → 🟡
      0% → ❌
    """
    c = max(0.0, min(1.0, float(coverage)))
    if c == 1.0:
        emoji = "🟩"
    elif c > 0.0:
        emoji = "🟡"
    else:
        emoji = "❌"
    return f"{emoji} {text}"
