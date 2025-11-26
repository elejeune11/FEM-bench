from fem_bench.pipeline_utils import FEMBenchPipeline
from llm_api.llm_clients import call_llm_for_code, call_llm_for_tests
from pathlib import Path
import json
from datetime import datetime

# === Config ===
# Use the SAME tasks/prompts for A and B (fair A/B test)
TASKS_DIR = "tasks"
PROMPTS_DIR = "prompts"

LLM_OUTPUTS_DIR = "llm_outputs"
RESULTS_DIR = "results"

PROMPT_TEMPLATE_DIR = "prompt_templates"
CODE_PROMPT_TEMPLATE_NAME = "code_prompt.j2"
TEST_PROMPT_TEMPLATE_NAME = "test_prompt.j2"

MODEL_NAMES = [
    "gemini-2.5-pro",
    "gemini-3-pro-preview"
]

SEED = 11
TEMPERATURE = 1.0

# === Setup pipeline ===
pipeline = FEMBenchPipeline(
    tasks_dir=TASKS_DIR,
    prompts_dir=PROMPTS_DIR,
    llm_outputs_dir=LLM_OUTPUTS_DIR,
    results_dir=RESULTS_DIR,
    prompt_template_dir=PROMPT_TEMPLATE_DIR,
    code_prompt_template_name=CODE_PROMPT_TEMPLATE_NAME,
    test_prompt_template_name=TEST_PROMPT_TEMPLATE_NAME
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
                        temperature=TEMPERATURE,           # NEW
                    )
                    code_path.write_text(code_out, encoding="utf-8")
                    print(f"      [✓] Code saved to: {code_path}")
                except Exception as e:
                    print(f"      [i] Writing marker (blocked/error) for CODE: {e}")

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
                        temperature=TEMPERATURE,           # NEW
                    )
                    test_out = "\n\n".join(test_out_dict.values())
                    test_path.write_text(test_out, encoding="utf-8")
                    print(f"      [✓] Tests saved to: {test_path}")
                except Exception as e:
                    print(f"      [i] Writing marker (blocked/error) for TESTS: {e}")

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
    "models": MODEL_NAMES,
    "seed": SEED,
    "temperature": TEMPERATURE,               # NEW
    "tasks_dir": TASKS_DIR,
    "prompts_dir": PROMPTS_DIR,
    "llm_outputs_dir": LLM_OUTPUTS_DIR,
    "results_dir": RESULTS_DIR,
}
(Path(RESULTS_DIR) / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

print("Pipeline complete.")
print(f"Models evaluated: {', '.join(MODEL_NAMES)}")