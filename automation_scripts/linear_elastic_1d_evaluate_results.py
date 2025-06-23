from fem_bench.pipeline_utils import FEMBenchPipeline
from pathlib import Path


overall_dir = Path(__file__).parent.parent

tasks_directory = overall_dir.joinpath("tasks").joinpath("tier1")

environment_file = overall_dir.joinpath("environments").joinpath("tier1_environment.yaml")

llm_outputs_directory = overall_dir.joinpath("LLM_query_results").joinpath("Gemini_2pt0_Flash_lite")

pipeline = FEMBenchPipeline(tasks_directory=tasks_directory, environment_file=environment_file, llm_outputs_directory=llm_outputs_directory)

pipeline.load_tasks()
pipeline.load_environment()

prompts = pipeline.generate_prompts(save=False)

pipeline.load_llm_outputs()

pipeline.evaluate_all_tasks()

score = pipeline.compute_aggregate_score()

aa = 44