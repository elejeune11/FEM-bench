from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

def build_sorted_task_table_with_colors(
    results_dirs: List[str | Path],
    *,
    models: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    table_title: str = "Task-by-task Accuracy@K (best of seeds) — ordered easiest→hardest",
) -> str:
    """
    Build a Markdown table sorted from 'easiest' to 'hardest' using a strong tiebreaker:
      1) #models with >=1 success across seeds (desc)
      2) total successes across all (models × seeds) (desc)
      3) overall success rate across all (models × seeds) (desc)
      4) task_id (asc)

    Each cell shows: 'X/K ✓' where X = #seeds that were successful for that model on that task.

    The Task column is now labeled by **coverage** using GitHub-safe emojis:
        🟩 = all models (100%) had ≥1 success
        🟡 = some but not all models had ≥1 success
        ❌ = no model had a success

    Expects files named '{task_id}_eval_{llm}.json' with top-level key 'matches_reference'.
    """
    seeds_data = [_load_single_seed_eval(d) for d in results_dirs]
    K = len(seeds_data)

    # Discover universe
    all_tasks = set()
    all_models = set()
    for seed in seeds_data:
        all_tasks.update(seed.keys())
        for m in {mm for t in seed.values() for mm in t.keys()}:
            all_models.add(m)

    task_list = [t for t in (tasks if tasks is not None else sorted(all_tasks)) if t in all_tasks]
    model_list = [m for m in (models if models is not None else sorted(all_models)) if m in all_models]

    # Build per-task stats
    rows = []
    sort_keys = []  # for later sorting
    for task_id in task_list:
        # Count successes per model across seeds
        cell_strings = []
        models_with_any = 0
        total_successes = 0  # across all models × seeds

        for m in model_list:
            x = 0
            for seed in seeds_data:
                ok = bool(seed.get(task_id, {}).get(m, False))
                if ok:
                    x += 1
            total_successes += x
            any_ok = (x >= 1)
            if any_ok:
                models_with_any += 1
            cell_strings.append(f"{x}/{K} {'✓' if any_ok else '×'}")

        # Aggregate metrics
        denom_models = max(len(model_list), 1)
        coverage = models_with_any / denom_models                    # used for emoji label
        denom_all = max(len(model_list) * K, 1)
        task_rate = total_successes / denom_all                      # still used for tertiary tiebreak

        # Emoji label by coverage (GitHub-safe)
        task_cell = _color_task_label(task_id, coverage)

        # Record for sorting: negative for desc
        sort_keys.append((
            -models_with_any,        # 1) coverage (primary)
            -total_successes,        # 2) total successes
            -task_rate,              # 3) overall success rate
            task_id                  # 4) stable
        ))
        rows.append([task_cell] + cell_strings)

    # Sort rows by the strong ordering
    rows_sorted = [x for _, x in sorted(zip(sort_keys, rows), key=lambda p: p[0])]

    # Final summary row: Overall Accuracy@K per model (how many tasks had >=1 success)
    overall_counts = []
    for j, m in enumerate(model_list, start=1):
        y = 0
        for r in rows_sorted:
            cell = r[j]  # e.g., "5/6 ✓" or "0/6 ×"
            x_val = int(str(cell).split("/", 1)[0].strip()) if "/" in str(cell) else 0
            if x_val >= 1:
                y += 1
        overall_counts.append(f"{y}/{len(rows_sorted)}")

    headers = ["Task"] + model_list
    df = pd.DataFrame(rows_sorted, columns=headers)
    df.loc[len(df)] = ["**Overall Accuracy@{}**".format(K)] + overall_counts

    title = f"### {table_title}\n\n"
    return title + df.to_markdown(index=False) + "\n"


# ------------------------ internals ------------------------

def _load_single_seed_eval(results_dir: str | Path) -> Dict[str, Dict[str, bool]]:
    """
    Read one results directory and return:
        { task_id: { model_name: matches_reference_bool } }

    Expects files named "{task_id}_eval_{llm}.json" with a top-level `matches_reference` key.
    """
    out: Dict[str, Dict[str, bool]] = {}
    d = Path(results_dir)
    for p in d.glob("*_eval_*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        task_id, model_name = _parse_eval_filename(p.name)
        out.setdefault(task_id, {})[model_name] = bool(data.get("matches_reference", False))
    return out


def _parse_eval_filename(filename: str) -> Tuple[str, str]:
    # "{task_id}_eval_{llm}.json" -> (task_id, llm)
    stem = filename[:-5] if filename.endswith(".json") else filename
    left, model = stem.split("_eval_", 1)
    return left, model


def _color_task_label(task_id: str, coverage: float) -> str:
    """
    Emoji label by COVERAGE (fraction of models with >=1 success):
      1.0 → 🟩  |  (0,1) → 🟡  |  0.0 → ❌
    """
    c = max(0.0, min(1.0, float(coverage)))
    if c == 1.0:
        emoji = "🟩"
    elif c > 0.0:
        emoji = "🟡"
    else:
        emoji = "❌"
    return f"{emoji} {task_id}"
