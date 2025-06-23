"""
FEM-Bench Pipeline for batch processing and evaluation.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
from datetime import datetime

from fem_bench.yaml_load import Environment, Task, load_environment, load_all_tasks
from fem_bench.prompt import ParsedCode, parse_llm_json_output
from fem_bench.evaluate import TestQualityEvaluator
from fem_bench.evaluate import evaluate_single_task


@dataclass
class PipelineConfig:
    """Configuration for FEMBenchPipeline."""
    tasks_directory: Path
    environment_file: Path
    llm_outputs_directory: Optional[Path] = None
    output_directory: Path = Path("./results")
    include_prompts: bool = True
    save_intermediate: bool = False


class FEMBenchPipeline:
    """
    Complete pipeline for FEM-Bench task processing and evaluation.
    
    Handles loading tasks, generating prompts, processing LLM outputs,
    and running comprehensive evaluations with dependency resolution.
    """
    
    def __init__(
        self,
        tasks_directory: Union[str, Path],
        environment_file: Union[str, Path],
        llm_outputs_directory: Optional[Union[str, Path]] = None,
        output_directory: Union[str, Path] = Path("./results")
    ):
        """
        Initialize FEMBenchPipeline with basic configuration.
        
        Args:
            tasks_directory: Directory containing task YAML files
            environment_file: Path to environment YAML configuration
            llm_outputs_directory: Directory containing LLM JSON outputs (optional)
            output_directory: Directory to save results and generated prompts
            
        Raises:
            FileNotFoundError: If required directories/files don't exist
        """
        # Convert and store paths
        self.tasks_directory = Path(tasks_directory)
        self.environment_file = Path(environment_file)
        self.llm_outputs_directory = Path(llm_outputs_directory) if llm_outputs_directory else None
        self.output_directory = Path(output_directory)
        self.prompts_directory = self.output_directory / "prompts"
        
        # Validate required paths exist
        if not self.tasks_directory.exists():
            raise FileNotFoundError(f"Tasks directory not found: {self.tasks_directory}")
        
        if not self.environment_file.exists():
            raise FileNotFoundError(f"Environment file not found: {self.environment_file}")
        
        if self.llm_outputs_directory and not self.llm_outputs_directory.exists():
            raise FileNotFoundError(f"LLM outputs directory not found: {self.llm_outputs_directory}")
        
        # Create output directory if needed
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize state (populated by individual methods)
        self.environment: Optional[Environment] = None
        self.tasks: List[Task] = []
        self.llm_outputs: Dict[str, ParsedCode] = {}
        self.evaluation_results: List[Dict] = []
        self.evaluator: Optional[TestQualityEvaluator] = None
        self._llm_outputs_cache: Optional[Dict[str, ParsedCode]] = None
    
    def load_tasks(self) -> List[Task]:
        """
        Load all tasks from the tasks directory.
        
        Returns:
            List of loaded Task objects, sorted by task_id
            
        Raises:
            ValueError: If no valid tasks are found
            Exception: If task loading fails for other reasons
        """
        try:
            self.tasks = load_all_tasks(self.tasks_directory)
        except Exception as e:
            raise Exception(f"Failed to load tasks from {self.tasks_directory}: {e}")
        
        if not self.tasks:
            raise ValueError(f"No valid tasks found in {self.tasks_directory}")
        
        return self.tasks
    
    def load_environment(self) -> Environment:
        """
        Load environment configuration.
        
        Returns:
            Loaded Environment object
            
        Raises:
            Exception: If environment loading fails
        """
        try:
            self.environment = load_environment(self.environment_file)
        except Exception as e:
            raise Exception(f"Failed to load environment from {self.environment_file}: {e}")
        
        return self.environment
    
    def generate_prompts(self, save: bool = True) -> Dict[str, str]:
        """
        Generate prompts for all loaded tasks.
        
        Args:
            save: Whether to save prompts to files in prompts_directory
            
        Returns:
            Dictionary mapping task_id -> prompt_string
            
        Raises:
            ValueError: If tasks or environment not loaded
            Exception: If prompt generation fails
        """
        # Ensure tasks and environment are loaded
        if not self.tasks:
            raise ValueError("No tasks loaded. Call load_tasks() first.")
        
        if not self.environment:
            raise ValueError("Environment not loaded. Call load_environment() first.")
        
        prompts = {}
        
        try:
            for task in self.tasks:
                # Generate prompt using existing build_prompt function
                from fem_bench.prompt import build_prompt
                prompt = build_prompt(task, self.environment, self.tasks_directory)
                prompts[task.task_id] = prompt
                
                # Save to file if requested
                if save:
                    self._save_prompt(task.task_id, prompt)
        
        except Exception as e:
            raise Exception(f"Failed to generate prompts: {e}")
        
        return prompts
    
    def _save_prompt(self, task_id: str, prompt: str) -> None:
        """Save a prompt to file."""
        # Create prompts directory if it doesn't exist
        self.prompts_directory.mkdir(parents=True, exist_ok=True)
        
        # Save prompt to file
        prompt_file = self.prompts_directory / f"{task_id}_prompt.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
    
    def load_llm_outputs(self) -> Dict[str, ParsedCode]:
        """
        Load LLM outputs from JSON files in llm_outputs_directory.
        
        Returns:
            Dictionary mapping task_id -> ParsedCode object
            
        Raises:
            ValueError: If llm_outputs_directory not configured or no outputs found
            Exception: If loading fails
        """
        if not self.llm_outputs_directory:
            raise ValueError("llm_outputs_directory not configured. Set it during initialization.")
        
        self.llm_outputs = {}
        
        try:
            # Find all JSON files in the outputs directory
            json_files = list(self.llm_outputs_directory.glob("*.json"))
            
            for json_file in json_files:
                # Extract task_id from filename (remove .json extension)
                task_id = json_file.stem
                
                # Parse the JSON file
                from fem_bench.prompt import parse_llm_json_output
                parsed_output = parse_llm_json_output(json_file)
                self.llm_outputs[task_id] = parsed_output
                
        except Exception as e:
            raise Exception(f"Failed to load LLM outputs from {self.llm_outputs_directory}: {e}")
        
        if not self.llm_outputs:
            raise ValueError(f"No valid LLM outputs found in {self.llm_outputs_directory}")
        
        return self.llm_outputs
    
    def evaluate_all_tasks(self, save_individual: bool = True) -> List[Dict]:
        """
        Evaluate all tasks and generate evaluation results.
        
        Args:
            save_individual: Whether to save individual task results as JSON files
            
        Returns:
            List of evaluation dictionaries (one per task)
            
        Raises:
            ValueError: If prerequisites not loaded
            Exception: If evaluation fails
        """
        # Ensure all prerequisites are loaded
        if not self.tasks:
            raise ValueError("No tasks loaded. Call load_tasks() first.")
        
        if not self.environment:
            raise ValueError("Environment not loaded. Call load_environment() first.")
        
        if not self.llm_outputs:
            raise ValueError("No LLM outputs loaded. Call load_llm_outputs() first.")
        
        # Initialize evaluator if needed
        if not self.evaluator:
            self.evaluator = TestQualityEvaluator(self.environment)
        
        self.evaluation_results = []
        
        try:
            for task in self.tasks:
                # Check if we have LLM output for this task
                if task.task_id not in self.llm_outputs:
                    # Create a failure result for missing LLM output
                    result = {
                        "task_id": task.task_id,
                        "fcn_correct_with_reference_provided": 0,
                        "fcn_correct_with_llm_chain": 0,
                        "total_tests": 0,
                        "tests_passed_on_reference": 0,
                        "total_expected_failures": 0,
                        "expected_failures_failed_on_reference": 0,
                        "error": f"No LLM output found for task {task.task_id}"
                    }
                else:
                    # Run evaluation using existing function
                    result = evaluate_single_task(
                        task=task,
                        environment=self.environment,
                        llm_output=self.llm_outputs[task.task_id],
                        task_dir=self.tasks_directory,
                        llm_outputs=self.llm_outputs
                    )
                
                self.evaluation_results.append(result)
                
                # Save individual result if requested
                if save_individual:
                    self._save_individual_result(task.task_id, result)
        
        except Exception as e:
            raise Exception(f"Failed to evaluate tasks: {e}")
        
        return self.evaluation_results
    
    def _save_individual_result(self, task_id: str, result: Dict) -> None:
        """Save individual task evaluation result to JSON file."""
        # Create results directory if it doesn't exist
        results_dir = self.output_directory / "individual_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save result to file
        result_file = results_dir / f"{task_id}_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
    
    def compute_aggregate_score(self) -> Dict[str, float]:
        """
        Compute aggregate scores from evaluation results.
        
        Returns:
            Dictionary with four aggregate metrics:
            - fcn_correct_with_reference_provided_pct: Percentage of tasks with correct functions (reference deps)
            - fcn_correct_with_llm_chain_pct: Percentage of tasks with correct functions (LLM deps)
            - avg_tests_passed_on_reference_pct: Average percentage of tests passed on reference
            - avg_expected_failures_detected_pct: Average percentage of expected failures detected
            
        Raises:
            ValueError: If no evaluation results available
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Call evaluate_all_tasks() first.")
        
        # Count function correctness
        total_tasks = len(self.evaluation_results)
        fcn_correct_ref = sum(1 for r in self.evaluation_results if r.get("fcn_correct_with_reference_provided", 0) == 1)
        fcn_correct_llm = sum(1 for r in self.evaluation_results if r.get("fcn_correct_with_llm_chain", 0) == 1)
        
        # Calculate test pass rates (skip tasks with 0 total tests)
        test_pass_rates = []
        for result in self.evaluation_results:
            total_tests = result.get("total_tests", 0)
            if total_tests > 0:
                tests_passed = result.get("tests_passed_on_reference", 0)
                test_pass_rates.append(tests_passed / total_tests)
        
        # Calculate failure detection rates (skip tasks with 0 expected failures)
        failure_detection_rates = []
        for result in self.evaluation_results:
            total_expected_failures = result.get("total_expected_failures", 0)
            if total_expected_failures > 0:
                failures_detected = result.get("expected_failures_failed_on_reference", 0)
                failure_detection_rates.append(failures_detected / total_expected_failures)
        
        # Compute averages
        avg_test_pass_rate = sum(test_pass_rates) / len(test_pass_rates) if test_pass_rates else 0.0
        avg_failure_detection_rate = sum(failure_detection_rates) / len(failure_detection_rates) if failure_detection_rates else 0.0
        
        return {
            "fcn_correct_with_reference_provided_pct": (fcn_correct_ref / total_tasks) * 100.0,
            "fcn_correct_with_llm_chain_pct": (fcn_correct_llm / total_tasks) * 100.0,
            "avg_tests_passed_on_reference_pct": avg_test_pass_rate * 100.0,
            "avg_expected_failures_detected_pct": avg_failure_detection_rate * 100.0
        }