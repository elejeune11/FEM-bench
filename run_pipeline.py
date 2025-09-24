from fem_bench.pipeline_utils import FEMBenchPipeline
from llm_api.llm_clients import call_llm_for_code, call_llm_for_tests
from pathlib import Path

# === Config ===
TASKS_DIR = "tasks"
PROMPTS_DIR = "prompts"
LLM_OUTPUTS_DIR = "llm_outputs"
RESULTS_DIR = "results"
MODEL_NAMES = [
    "gpt-4o",
    "gpt-5",
    "gemini-2.5-pro",
    "claude-3-5",
    "claude-sonnet-4",
    "claude-opus-4.1",
    "deepseek-chat",
]
SEED = 11

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
Path(LLM_OUTPUTS_DIR).mkdir(exist_ok=True)

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
                    code_out = call_llm_for_code(model_name, code_prompt, seed=SEED)
                    code_path.write_text(code_out)
                    print(f"      [✓] Code saved to: {code_path}")
                except Exception as e:
                    print(f"      [Error] Failed code gen for {task_id}: {e}")

        # --- Test Prompt ---
        test_prompt = prompt_pair.get("tests")
        if test_prompt:
            test_path = Path(LLM_OUTPUTS_DIR) / f"{task_id}_test_{model_name}.py"
            if test_path.exists():
                print(f"      [✓] Skipping tests (already exists): {test_path}")
            else:
                print(f"    [→] Generating TESTS for task: {task_id}")
                try:
                    test_out_dict = call_llm_for_tests(model_name, test_prompt, seed=SEED)
                    test_out = "\n\n".join(test_out_dict.values())
                    test_path.write_text(test_out)
                    print(f"      [✓] Tests saved to: {test_path}")
                except Exception as e:
                    print(f"      [Error] Failed test gen for {task_id}: {e}")

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

print("Pipeline complete.")
print(f"Models evaluated: {', '.join(MODEL_NAMES)}")
