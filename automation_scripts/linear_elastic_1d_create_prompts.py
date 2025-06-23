from fem_bench.pipeline_utils import FEMBenchPipeline
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

overall_dir = Path(__file__).parent.parent

tasks_directory = overall_dir.joinpath("tasks").joinpath("tier1")

environment_file = overall_dir.joinpath("environments").joinpath("tier1_environment.yaml")

pipeline = FEMBenchPipeline(tasks_directory=tasks_directory, environment_file=environment_file)

pipeline.load_tasks()
pipeline.load_environment()

prompts = pipeline.generate_prompts(save=False)


aa = 44