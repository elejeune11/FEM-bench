"""
Test for FEMBenchPipeline initialization.
"""
import json
import pytest
import tempfile
import shutil
from pathlib import Path

from fem_bench.pipeline_utils import FEMBenchPipeline, PipelineConfig


"""
Test for FEMBenchPipeline initialization.
"""
import pytest
import tempfile
import shutil
from pathlib import Path

from fem_bench.pipeline_utils import FEMBenchPipeline


@pytest.fixture
def test_paths():
    """Set up test paths using example files."""
    test_dir = Path(__file__).parent
    temp_dir = Path(tempfile.mkdtemp())
    
    # Use example directories from the tests folder
    tasks_dir = test_dir / "example_tasks"
    outputs_dir = test_dir / "example_results" 
    env_file = test_dir / "example_environments" / "simple_environment.yaml"
    
    # Verify example files exist
    if not tasks_dir.exists():
        pytest.skip(f"Example tasks directory not found: {tasks_dir}")
    if not env_file.exists():
        pytest.skip(f"Example environment file not found: {env_file}")
    
    yield {
        'temp_dir': temp_dir,
        'tasks_dir': tasks_dir,
        'outputs_dir': outputs_dir,
        'env_file': env_file
    }
    
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def test_basic_initialization(test_paths):
    """Test basic pipeline initialization."""
    pipeline = FEMBenchPipeline(
        tasks_directory=test_paths['tasks_dir'],
        environment_file=test_paths['env_file']
    )
    
    # Check paths are set correctly
    assert pipeline.tasks_directory == test_paths['tasks_dir']
    assert pipeline.environment_file == test_paths['env_file']
    assert pipeline.llm_outputs_directory is None
    
    # Check state is uninitialized
    assert pipeline.environment is None
    assert pipeline.tasks == []


def test_initialization_with_llm_outputs(test_paths):
    """Test initialization with LLM outputs directory."""
    pipeline = FEMBenchPipeline(
        tasks_directory=test_paths['tasks_dir'],
        environment_file=test_paths['env_file'],
        llm_outputs_directory=test_paths['outputs_dir']
    )
    
    assert pipeline.llm_outputs_directory == test_paths['outputs_dir']


def test_missing_required_paths_raise_errors(test_paths):
    """Test that missing required paths raise FileNotFoundError."""
    missing_dir = test_paths['temp_dir'] / "missing"
    
    # Missing tasks directory
    with pytest.raises(FileNotFoundError, match="Tasks directory not found"):
        FEMBenchPipeline(
            tasks_directory=missing_dir,
            environment_file=test_paths['env_file']
        )
    
    # Missing environment file
    with pytest.raises(FileNotFoundError, match="Environment file not found"):
        FEMBenchPipeline(
            tasks_directory=test_paths['tasks_dir'],
            environment_file=missing_dir / "missing.yaml"
        )


def test_load_tasks(test_paths):
    """Test loading tasks from directory."""
    pipeline = FEMBenchPipeline(
        tasks_directory=test_paths['tasks_dir'],
        environment_file=test_paths['env_file']
    )
    
    # Initially no tasks loaded
    assert pipeline.tasks == []
    
    # Load tasks
    tasks = pipeline.load_tasks()
    
    # Should have loaded tasks
    assert len(tasks) > 0
    assert len(pipeline.tasks) > 0
    assert pipeline.tasks == tasks
    
    # Tasks should be sorted by task_id
    task_ids = [task.task_id for task in tasks]
    assert task_ids == sorted(task_ids)


def test_load_tasks_empty_directory(test_paths):
    """Test loading tasks from empty directory raises ValueError."""
    empty_dir = test_paths['temp_dir'] / "empty_tasks"
    empty_dir.mkdir()
    
    pipeline = FEMBenchPipeline(
        tasks_directory=empty_dir,
        environment_file=test_paths['env_file']
    )
    
    with pytest.raises(ValueError, match="No valid tasks found"):
        pipeline.load_tasks()


def test_generate_prompts(test_paths):
    """Test generating prompts for all tasks."""
    pipeline = FEMBenchPipeline(
        tasks_directory=test_paths['tasks_dir'],
        environment_file=test_paths['env_file']
    )
    
    # Load prerequisites
    pipeline.load_tasks()
    pipeline.load_environment()
    
    # Generate prompts without saving
    prompts = pipeline.generate_prompts(save=False)
    
    # Should have prompts for all tasks
    assert len(prompts) == len(pipeline.tasks)
    
    # Check prompt content
    for task in pipeline.tasks:
        assert task.task_id in prompts
        prompt = prompts[task.task_id]
        assert len(prompt) > 0
        assert task.title in prompt
        assert task.expected_function_name in prompt


def test_generate_prompts_saves_files(test_paths):
    """Test that generate_prompts saves files when save=True."""
    pipeline = FEMBenchPipeline(
        tasks_directory=test_paths['tasks_dir'],
        environment_file=test_paths['env_file'],
        output_directory=test_paths['temp_dir'] / "output"
    )
    
    # Load prerequisites
    pipeline.load_tasks()
    pipeline.load_environment()
    
    # Generate and save prompts
    prompts = pipeline.generate_prompts(save=True)
    
    # Check files were created
    assert pipeline.prompts_directory.exists()
    
    for task_id in prompts:
        prompt_file = pipeline.prompts_directory / f"{task_id}_prompt.txt"
        assert prompt_file.exists()
        
        # Verify file content matches
        with open(prompt_file, 'r', encoding='utf-8') as f:
            file_content = f.read()
        assert file_content == prompts[task_id]


def test_generate_prompts_requires_prerequisites(test_paths):
    """Test that generate_prompts requires tasks and environment to be loaded."""
    pipeline = FEMBenchPipeline(
        tasks_directory=test_paths['tasks_dir'],
        environment_file=test_paths['env_file']
    )
    
    # Should fail without tasks loaded
    with pytest.raises(ValueError, match="No tasks loaded"):
        pipeline.generate_prompts()
    
    # Load tasks but not environment
    pipeline.load_tasks()
    
    # Should fail without environment loaded
    with pytest.raises(ValueError, match="Environment not loaded"):
        pipeline.generate_prompts()


def test_load_llm_outputs(test_paths):
    """Test loading LLM outputs from JSON files."""
    pipeline = FEMBenchPipeline(
        tasks_directory=test_paths['tasks_dir'],
        environment_file=test_paths['env_file'],
        llm_outputs_directory=test_paths['outputs_dir']
    )
    
    # Initially no outputs loaded
    assert pipeline.llm_outputs == {}
    
    # Load outputs
    outputs = pipeline.load_llm_outputs()
    
    # Should have loaded outputs
    assert len(outputs) > 0
    assert len(pipeline.llm_outputs) > 0
    assert pipeline.llm_outputs == outputs
    
    # Check that outputs are ParsedCode objects
    for task_id, parsed_code in outputs.items():
        assert hasattr(parsed_code, 'main_function')
        assert hasattr(parsed_code, 'function_imports')


def test_load_llm_outputs_no_directory_configured(test_paths):
    """Test that load_llm_outputs fails if no llm_outputs_directory configured."""
    pipeline = FEMBenchPipeline(
        tasks_directory=test_paths['tasks_dir'],
        environment_file=test_paths['env_file']
        # No llm_outputs_directory provided
    )
    
    with pytest.raises(ValueError, match="llm_outputs_directory not configured"):
        pipeline.load_llm_outputs()


def test_evaluate_all_tasks(test_paths):
    """Test evaluating all tasks."""
    pipeline = FEMBenchPipeline(
        tasks_directory=test_paths['tasks_dir'],
        environment_file=test_paths['env_file'],
        llm_outputs_directory=test_paths['outputs_dir'],
        output_directory=test_paths['temp_dir'] / "output"
    )
    
    # Load all prerequisites
    pipeline.load_tasks()
    pipeline.load_environment()
    pipeline.load_llm_outputs()
    
    # Initially no results
    assert pipeline.evaluation_results == []
    
    # Evaluate all tasks without saving individual files
    results = pipeline.evaluate_all_tasks(save_individual=False)
    
    # Should have results for all tasks
    assert len(results) > 0
    assert len(pipeline.evaluation_results) > 0
    assert pipeline.evaluation_results == results
    
    # Check result structure
    for result in results:
        assert "task_id" in result
        assert "fcn_correct_with_reference_provided" in result
        assert "fcn_correct_with_llm_chain" in result
        assert "total_tests" in result
        assert "tests_passed_on_reference" in result
        assert "total_expected_failures" in result
        assert "expected_failures_failed_on_reference" in result


def test_evaluate_all_tasks_saves_individual_files(test_paths):
    """Test that evaluate_all_tasks saves individual JSON files."""
    pipeline = FEMBenchPipeline(
        tasks_directory=test_paths['tasks_dir'],
        environment_file=test_paths['env_file'],
        llm_outputs_directory=test_paths['outputs_dir'],
        output_directory=test_paths['temp_dir'] / "output"
    )
    
    # Load prerequisites
    pipeline.load_tasks()
    pipeline.load_environment()
    pipeline.load_llm_outputs()
    
    # Evaluate and save individual files
    results = pipeline.evaluate_all_tasks(save_individual=True)
    
    # Check individual files were created
    results_dir = pipeline.output_directory / "individual_results"
    assert results_dir.exists()
    
    for result in results:
        task_id = result["task_id"]
        result_file = results_dir / f"{task_id}_result.json"
        assert result_file.exists()
        
        # Verify file content matches
        with open(result_file, 'r', encoding='utf-8') as f:
            file_content = json.load(f)
        assert file_content == result


def test_evaluate_all_tasks_requires_prerequisites(test_paths):
    """Test that evaluate_all_tasks requires all prerequisites to be loaded."""
    pipeline = FEMBenchPipeline(
        tasks_directory=test_paths['tasks_dir'],
        environment_file=test_paths['env_file'],
        llm_outputs_directory=test_paths['outputs_dir']
    )
    
    # Should fail without tasks loaded
    with pytest.raises(ValueError, match="No tasks loaded"):
        pipeline.evaluate_all_tasks()
    
    # Load tasks but not environment
    pipeline.load_tasks()
    with pytest.raises(ValueError, match="Environment not loaded"):
        pipeline.evaluate_all_tasks()
    
    # Load environment but not LLM outputs
    pipeline.load_environment()
    with pytest.raises(ValueError, match="No LLM outputs loaded"):
        pipeline.evaluate_all_tasks()


def test_compute_aggregate_score(test_paths):
    """Test computing aggregate scores from evaluation results."""
    pipeline = FEMBenchPipeline(
        tasks_directory=test_paths['tasks_dir'],
        environment_file=test_paths['env_file'],
        llm_outputs_directory=test_paths['outputs_dir']
    )
    
    # Load prerequisites and evaluate
    pipeline.load_tasks()
    pipeline.load_environment()
    pipeline.load_llm_outputs()
    pipeline.evaluate_all_tasks(save_individual=False)
    
    # Compute aggregate score
    scores = pipeline.compute_aggregate_score()
    
    # Check structure
    expected_keys = {
        "fcn_correct_with_reference_provided_pct",
        "fcn_correct_with_llm_chain_pct", 
        "avg_tests_passed_on_reference_pct",
        "avg_expected_failures_detected_pct"
    }
    assert set(scores.keys()) == expected_keys
    
    # Check values are percentages (0-100)
    for key, value in scores.items():
        assert isinstance(value, float)
        assert 0.0 <= value <= 100.0


def test_compute_aggregate_score_requires_evaluation_results(test_paths):
    """Test that compute_aggregate_score requires evaluation results."""
    pipeline = FEMBenchPipeline(
        tasks_directory=test_paths['tasks_dir'],
        environment_file=test_paths['env_file']
    )
    
    with pytest.raises(ValueError, match="No evaluation results available"):
        pipeline.compute_aggregate_score()


def test_compute_aggregate_score_handles_edge_cases(test_paths):
    """Test aggregate score computation with edge cases."""
    pipeline = FEMBenchPipeline(
        tasks_directory=test_paths['tasks_dir'],
        environment_file=test_paths['env_file']
    )
    
    # Mock evaluation results with edge cases (bypassing normal loading)
    pipeline.evaluation_results = [
        {
            "task_id": "test1",
            "fcn_correct_with_reference_provided": 1,
            "fcn_correct_with_llm_chain": 0,
            "total_tests": 0,  # No tests
            "tests_passed_on_reference": 0,
            "total_expected_failures": 0,  # No expected failures
            "expected_failures_failed_on_reference": 0
        },
        {
            "task_id": "test2", 
            "fcn_correct_with_reference_provided": 0,
            "fcn_correct_with_llm_chain": 1,
            "total_tests": 4,
            "tests_passed_on_reference": 2,  # 50% pass rate
            "total_expected_failures": 3,
            "expected_failures_failed_on_reference": 1  # 33% detection rate
        }
    ]
    
    scores = pipeline.compute_aggregate_score()
    
    # Function correctness: 1/2 = 50% for ref, 1/2 = 50% for llm
    assert scores["fcn_correct_with_reference_provided_pct"] == 50.0
    assert scores["fcn_correct_with_llm_chain_pct"] == 50.0
    
    # Test pass rate: only test2 has tests, 2/4 = 50%
    assert scores["avg_tests_passed_on_reference_pct"] == 50.0
    
    # Failure detection: only test2 has expected failures, 1/3 ≈ 33.33%
    assert abs(scores["avg_expected_failures_detected_pct"] - 33.333333333333336) < 1e-10