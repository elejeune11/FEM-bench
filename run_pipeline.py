from fem_bench.pipeline_utils import FEMBenchPipeline
from llm_api.llm_clients import call_llm_for_code, call_llm_for_tests
from pathlib import Path
import json
from datetime import datetime

# === Config ===
# Use the SAME tasks/prompts for A and B (fair A/B test)
TASKS_DIR = "tasks"
PROMPTS_DIR = "prompts"

with_system_prompt = True

if with_system_prompt:
    LLM_OUTPUTS_DIR = "llm_outputs_system_prompt_v2"
    RESULTS_DIR = "results_system_prompt_v2"
#     SYSTEM_PROMPT = """ v1
# You are an expert Python engineer in scientific computing, with a specialization in Finite Element Analysis (FEM).
# Before responding, reason privately to ensure mathematical correctness and edge-case coverage, but do not output your reasoning.
# Always return only executable Python code—no commentary, markdown fences, or extra text.
# Follow the user’s instructions exactly; if there is any conflict, prefer the user’s explicit task rules.
# Never add imports beyond those specified, and never reimplement or alter helper functions.
# If the task is to write tests, output only pytest test functions.
# Be precise, deterministic, and correctness-focused.
# """
    SYSTEM_PROMPT = """
You are an expert in finite element analysis and scientific computing. You completed your PhD under Tom Hughes and have spent over 10 years at Sandia National Laboratories working on computational mechanics problems.
Focus on producing robust, correct, production-quality Python code. Your solutions should demonstrate both mathematical rigor and practical engineering judgment.
Output only executable Python code—no markdown, comments, or extra text.
Follow the user's task rules exactly: match the given function signatures and docstrings, respect import limits, and never alter helper functions.
If the task is to write tests, output only pytest tests with meaningful assertions.
"""
else:
    LLM_OUTPUTS_DIR = "llm_outputs"
    RESULTS_DIR = "results"
    SYSTEM_PROMPT = None  # no system message

# Convenience: single variable to pass to clients
ACTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT

MODEL_NAMES = [
    "gpt-4o",
    "gpt-5",
    "gemini-1.5-flash",
    "gemini-2.5-pro",
    "claude-3-5",
    "claude-sonnet-4",
    "claude-opus-4.1",
    "deepseek-chat",
    "deepseek-reasoner",
]
SEED = 11

# --- Helper: write a simple marker file so you can see why it failed/was blocked ---
def _write_block_marker(py_path: Path, *, provider: str, task_id: str, phase: str, error: Exception) -> None:
    reason = str(error).replace("\n", " ")[:2000]  # keep it readable
    lines = [
        "# BLOCKED_OR_ERROR",
        f"# Provider: {provider}",
        f"# Phase: {phase}",                 # CODE or TESTS
        f"# Task: {task_id}",
        f"# Reason: {reason}",
        "# Note: No code generated due to provider block or error.",
        "",
    ]
    py_path.write_text("\n".join(lines), encoding="utf-8")

# === Setup pipeline ===
pipeline = FEMBenchPipeline(
    tasks_dir=TASKS_DIR,
    prompts_dir=PROMPTS_DIR,
    llm_outputs_dir=LLM_OUTPUTS_DIR,
    results_dir=RESULTS_DIR,
)

# === 1. Load tasks ===
print("[1] Loading tasks...")
pipeline.load_all_tasks()

# === 2. Generate prompts ===
print("[2] Generating prompts...")
pipeline.generate_and_save_task_prompts()
pipeline.generate_and_save_test_prompts()

# === 3. Generate completions for each model ===
print("[3] Generating code/test completions...")
Path(LLM_OUTPUTS_DIR).mkdir(exist_ok=True, parents=True)

for model_name in MODEL_NAMES:
    print(f"--- Generating outputs for model: {model_name} ---")

    for task_id, prompt_pair in pipeline.prompts.items():
        # --- Code Prompt ---
        code_prompt = prompt_pair.get("code")
        if code_prompt:
            code_path = Path(LLM_OUTPUTS_DIR) / f"{task_id}_code_{model_name}.py"
            if code_path.exists():
                print(f"      [✓] Skipping code (already exists): {code_path}")
            else:
                print(f"    [→] Generating CODE for task: {task_id}")
                try:
                    code_out = call_llm_for_code(
                        model_name,
                        code_prompt,
                        seed=SEED,
                        system_prompt=ACTIVE_SYSTEM_PROMPT,  # NEW
                    )
                    code_path.write_text(code_out, encoding="utf-8")
                    print(f"      [✓] Code saved to: {code_path}")
                except Exception as e:
                    print(f"      [i] Writing marker (blocked/error) for CODE: {e}")
                    _write_block_marker(code_path, provider=model_name, task_id=task_id, phase="CODE", error=e)

        # --- Test Prompt ---
        test_prompt = prompt_pair.get("tests")
        if test_prompt:
            test_path = Path(LLM_OUTPUTS_DIR) / f"{task_id}_test_{model_name}.py"
            if test_path.exists():
                print(f"      [✓] Skipping tests (already exists): {test_path}")
            else:
                print(f"    [→] Generating TESTS for task: {task_id}")
                try:
                    test_out_dict = call_llm_for_tests(
                        model_name,
                        test_prompt,
                        seed=SEED,
                        system_prompt=ACTIVE_SYSTEM_PROMPT,  # NEW
                    )
                    test_out = "\n\n".join(test_out_dict.values())
                    test_path.write_text(test_out, encoding="utf-8")
                    print(f"      [✓] Tests saved to: {test_path}")
                except Exception as e:
                    print(f"      [i] Writing marker (blocked/error) for TESTS: {e}")
                    _write_block_marker(test_path, provider=model_name, task_id=task_id, phase="TESTS", error=e)

# === 4. Load completions ===
print("[4] Loading LLM outputs...")
pipeline.load_all_llm_outputs(allowed_llms=MODEL_NAMES)

# === 5. Evaluate code + tests ===
print("[5] Evaluating generated functions...")
pipeline.evaluate_all_llm_outputs()

print("[6] Evaluating generated test files...")
pipeline.evaluate_all_llm_tests()

# === 7. Score + summary ===
print("[7] Aggregating results...")
pipeline.compute_aggregate_score()
pipeline.create_markdown_summary(model_names=MODEL_NAMES)

# Stamp run metadata for auditing
Path(RESULTS_DIR).mkdir(exist_ok=True, parents=True)
meta = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "with_system_prompt": with_system_prompt,
    "system_prompt": SYSTEM_PROMPT.strip() if SYSTEM_PROMPT else None,
    "models": MODEL_NAMES,
    "seed": SEED,
    "tasks_dir": TASKS_DIR,
    "prompts_dir": PROMPTS_DIR,
    "llm_outputs_dir": LLM_OUTPUTS_DIR,
    "results_dir": RESULTS_DIR,
}
(Path(RESULTS_DIR) / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

print("Pipeline complete.")
print(f"Models evaluated: {', '.join(MODEL_NAMES)}")
