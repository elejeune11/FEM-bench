from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# ------- Public API -----------------------------------------------------------

def build_task_by_task_accuracy_at_k_markdown(
    results_dirs: List[str | Path],
    *,
    models: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    k: Optional[int] = None,     # if None, use number of seeds (=len(results_dirs))
    sort_tasks: bool = True,
    table_title: Optional[str] = None,
) -> str:
    """
    Aggregate multiple seed runs and render a task-by-task Accuracy@k table.

    Each cell shows "X/S ✓" where:
      - S = number of seeds (len(results_dirs)) unless k is given
      - X = count of seeds where the model solved the task (matches_reference=True)
      - ✓ if X >= 1 else ×

    The final row shows Overall Accuracy@k as "Y/|tasks|", where Y is
    the number of tasks solved at least once across seeds for each model.

    Parameters
    ----------
    results_dirs : list[str|Path]
        Each directory should contain JSON files named like "{task_id}_eval_{llm}.json".
    models : optional list[str]
        Restrict/ordering of model columns (e.g., ["gpt-4o","gpt-5",...]).
        If None, inferred from files (sorted).
    tasks : optional list[str]
        Restrict/ordering of task rows. If None, inferred from files (sorted).
    k : optional int
        The @k definition. If None, defaults to number of seeds (len(results_dirs)).
        This function does "best-of-seeds" (any success); k is used only for label clarity.
    sort_tasks : bool
        If True and tasks not provided, sort tasks alphabetically.
    table_title : optional str
        Optional first-level title string.

    Returns
    -------
    markdown : str
        A copy/paste-ready Markdown table with a final "Overall Accuracy@k" row.
    """
    # 1) Load results for each seed directory
    seeds_data = [ _load_single_seed(d) for d in results_dirs ]
    seed_count = len(seeds_data)
    K = k or seed_count

    # 2) Universe of tasks/models
    all_tasks = set()
    all_models = set()
    for seed in seeds_data:
        all_tasks.update(seed.keys())
        for t in seed.values():
            all_models.update(t.keys())

    if tasks is None:
        task_list = sorted(all_tasks) if sort_tasks else list(all_tasks)
    else:
        task_list = [t for t in tasks if t in all_tasks]

    if models is None:
        model_list = sorted(all_models)
    else:
        model_list = [m for m in models if m in all_models]

    # 3) Count successes per (task, model) across seeds
    #    success = any seed has matches_reference == True for that (task, model)
    per_task_rows: List[List[str]] = []
    overall_counts = {m: 0 for m in model_list}

    for task_id in task_list:
        row = [task_id]
        for m in model_list:
            successes = 0
            for seed in seeds_data:
                if task_id in seed and m in seed[task_id]:
                    if seed[task_id][m]:  # True means matches_reference
                        successes += 1
            mark = "✓" if successes >= 1 else "×"
            if successes >= 1:
                overall_counts[m] += 1
            row.append(f"{successes}/{seed_count} {mark}")
        per_task_rows.append(row)

    # 4) Final Overall Accuracy@K row
    final_row = [f"**Overall Accuracy@{K}**"]
    for m in model_list:
        solved = overall_counts[m]
        final_row.append(f"{solved}/{len(task_list)}")
    per_task_rows.append(final_row)

    # 5) Build markdown via pandas
    headers = ["Task"] + model_list
    df = pd.DataFrame(per_task_rows, columns=headers)

    title = f"### {table_title}\n\n" if table_title else ""
    return title + df.to_markdown(index=False) + "\n"


# ------- Internals ------------------------------------------------------------

def _load_single_seed(results_dir: str | Path) -> Dict[str, Dict[str, bool]]:
    """
    Read one results directory and return:
        { task_id: { model_name: matches_reference_bool } }

    Expects files named "{task_id}_eval_{llm}.json" with a top-level `matches_reference` key.
    """
    results: Dict[str, Dict[str, bool]] = {}
    results_dir = Path(results_dir)
    for p in results_dir.glob("*_eval_*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        task_id, model_name = _parse_eval_filename(p.name)
        matches = bool(data.get("matches_reference", False))
        results.setdefault(task_id, {})[model_name] = matches
    return results


def _parse_eval_filename(filename: str) -> Tuple[str, str]:
    """
    Parse "{task_id}_eval_{llm}.json" -> (task_id, llm).
    """
    # Split on "_eval_"
    stem = filename
    if stem.endswith(".json"):
        stem = stem[:-5]
    left, model = stem.split("_eval_", 1)
    return left, model
