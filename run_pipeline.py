# --- run_pipeline.py (updated) ---
from fem_bench.pipeline_utils import FEMBenchPipeline
from llm_api.openai_client import call_openai_chat, call_openai_for_tests
from pathlib import Path

# === Config ===
TASKS_DIR = "tasks"
PROMPTS_DIR = "prompts"
LLM_OUTPUTS_DIR = "llm_outputs"
RESULTS_DIR = "results"
LLM_NAME = "gpt4o"  # used in file naming
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

# === 3. Generate completions using OpenAI ===
print("[3] Generating code/test completions via OpenAI...")
Path(LLM_OUTPUTS_DIR).mkdir(exist_ok=True)

for task_id, prompt_pair in pipeline.prompts.items():
    # --- Code Prompt ---
    code_prompt = prompt_pair.get("code")
    if code_prompt:
        print(f"    [→] Generating CODE for task: {task_id}")
        try:
            code_out = call_openai_chat(code_prompt, seed=SEED)
            out_path = Path(LLM_OUTPUTS_DIR) / f"{task_id}_code_{LLM_NAME}.py"
            out_path.write_text(code_out)
            print(f"      [✓] Code saved to: {out_path}")
        except Exception as e:
            print(f"      [Error] Failed code gen for {task_id}: {e}")

    # --- Test Prompt ---
    test_prompt = prompt_pair.get("tests")
    if test_prompt:
        print(f"    [→] Generating TESTS for task: {task_id}")
        try:
            test_out_dict = call_openai_for_tests(test_prompt, seed=SEED)
            test_out = "\n\n".join(test_out_dict.values())
            out_path = Path(LLM_OUTPUTS_DIR) / f"{task_id}_test_{LLM_NAME}.py"
            out_path.write_text(test_out)
            print(f"      [✓] Tests saved to: {out_path}")
        except Exception as e:
            print(f"      [Error] Failed test gen for {task_id}: {e}")

# === 4. Load completions ===
print("[4] Loading LLM outputs...")
pipeline.load_all_llm_outputs(allowed_llms=[LLM_NAME])

# === 5. Evaluate code + tests ===
print("[5] Evaluating generated functions...")
pipeline.evaluate_all_llm_outputs()

print("[6] Evaluating generated test files...")
pipeline.evaluate_all_llm_tests()

# === 7. Score + summary ===
print("[7] Aggregating results...")
pipeline.compute_aggregate_score()
pipeline.create_markdown_summary()

print("Pipeline complete.")

aa = 44