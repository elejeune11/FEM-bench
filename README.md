# FEM-bench

## Setup Instructions

### Prerequisites
- Python 3.12

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd fem-bench
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Create virtual environment with Python 3.10 -- later versions of python should also work
   python3.10 -m venv fem_bench_env
   
   # Activate virtual environment
   # On Linux/Mac:
   source fem_bench_env/bin/activate
   
   # On Windows:
   fem_bench_env\Scripts\activate
   ```

3. **Install the package:**
   ```bash
   # Upgrade pip first (required for pyproject.toml support)
   pip install --upgrade pip
   
   # Install in editable mode with development dependencies
   pip install -e ".[dev]"
   
   # Or install just the package
   pip install -e .
   ```

4. **Verify installation:**
   ```bash
   python -c "import fem_bench; print('FEM-Bench installed successfully')"
   ```

5. **Additional Installs for Prompting:**
```bash
pip install selenium
```

### Running Tests
```bash
pytest tests/
```

### Deactivating Environment
```bash
deactivate
```

### Removing Environment
```bash
rm -rf fem_bench_env
```

## Workflow
FEM-bench evaluates both implementation ability and mathematical competency through a dual-task approach. The benchmark system reads YAML task definitions and generates clean prompts asking LLMs to both implement finite element functions (e.g., shape functions, numerical integration) and write comprehensive pytest tests validating the required mathematical properties (e.g., partition of unity, interpolation conditions). The system then scores both the correctness of the implementation against reference solutions and the quality of the generated tests against the theoretical requirements. This measures mathematical competency by evaluating whether models can identify and verify the core mathematical properties that ensure FEM solution correctness. The goal is to go beyond just pattern-matching code solutions.

## Using the FEMBenchPipeline

The `FEMBenchPipeline` class provides a complete automated workflow for generating prompts, evaluating LLM outputs, and computing aggregate scores across all tasks.

### Quick Start

Here's a minimal example to run the complete evaluation pipeline:

```python
from fem_bench.pipeline_utils import FEMBenchPipeline

# Initialize pipeline with your data directories
pipeline = FEMBenchPipeline(
    tasks_directory="tasks/tier1/",
    environment_file="environments/tier1_environment.yaml", 
    llm_outputs_directory="my_llm_outputs/",
    output_directory="results/"
)

# Load all required data
pipeline.load_tasks()
pipeline.load_environment()
pipeline.load_llm_outputs()

# Generate prompts for all tasks
prompts = pipeline.generate_prompts(save=True)

# Evaluate all tasks and save individual results
results = pipeline.evaluate_all_tasks(save_individual=True)

# Compute aggregate performance scores
scores = pipeline.compute_aggregate_score()

print(f"Function correctness (ref deps): {scores['fcn_correct_with_reference_provided_pct']:.1f}%")
print(f"Function correctness (LLM deps): {scores['fcn_correct_with_llm_chain_pct']:.1f}%")
print(f"Average test pass rate: {scores['avg_tests_passed_on_reference_pct']:.1f}%")
print(f"Average failure detection rate: {scores['avg_expected_failures_detected_pct']:.1f}%")
```

### Step-by-Step Usage

#### 1. Initialize the Pipeline

```python
from fem_bench.pipeline_utils import FEMBenchPipeline

pipeline = FEMBenchPipeline(
    tasks_directory="tasks/tier1/",           # Directory containing task YAML files
    environment_file="env/tier1_env.yaml",   # Environment configuration
    llm_outputs_directory="outputs/",        # Directory with LLM JSON responses (optional)
    output_directory="results/"              # Where to save generated files
)
```

The pipeline will automatically create subdirectories in `output_directory`:
- `results/prompts/` - Generated prompt files
- `results/individual_results/` - Individual task evaluation results

#### 2. Load Tasks and Environment

```python
# Load all task definitions from YAML files
tasks = pipeline.load_tasks()
print(f"Loaded {len(tasks)} tasks")

# Load environment configuration (libraries, requirements, etc.)
environment = pipeline.load_environment()
print(f"Environment: {environment.environment_name}")
```

#### 3. Generate Prompts

```python
# Generate prompts for all tasks
prompts = pipeline.generate_prompts(save=True)

# Access individual prompts
for task_id, prompt in prompts.items():
    print(f"Task {task_id}: {len(prompt)} characters")
    
# Prompts are automatically saved to results/prompts/{task_id}_prompt.txt
```

You can also generate prompts without saving files:

```python
prompts = pipeline.generate_prompts(save=False)
```

#### 4. Load LLM Outputs

If you have LLM responses to evaluate, load them:

```python
# Load LLM outputs from JSON files
llm_outputs = pipeline.load_llm_outputs()
print(f"Loaded outputs for {len(llm_outputs)} tasks")

# Access individual parsed outputs
for task_id, parsed_code in llm_outputs.items():
    print(f"Task {task_id}: {parsed_code.main_function_name}")
```

Expected JSON format for LLM outputs:
```json
{
    "function_imports": ["numpy", "scipy"],
    "test_imports": ["pytest"],
    "my_function": "def my_function(x: float) -> float:\n    return x * 2",
    "test_my_function": "def test_my_function():\n    assert my_function(2.0) == 4.0"
}
```

#### 5. Evaluate All Tasks

```python
# Run comprehensive evaluation on all tasks
results = pipeline.evaluate_all_tasks(save_individual=True)

print(f"Evaluated {len(results)} tasks")

# Each result contains detailed metrics
for result in results:
    task_id = result["task_id"]
    fcn_correct = result["fcn_correct_with_reference_provided"]
    test_quality = result["tests_passed_on_reference"] / max(1, result["total_tests"])
    print(f"{task_id}: Function correct={fcn_correct}, Test quality={test_quality:.2f}")
```

Individual results are saved as JSON files: `results/individual_results/{task_id}_result.json`

To evaluate without saving individual files:
```python
results = pipeline.evaluate_all_tasks(save_individual=False)
```

#### 6. Compute Aggregate Scores

```python
# Get overall performance metrics
scores = pipeline.compute_aggregate_score()

print("=== FEM-Bench Results ===")
print(f"Function Correctness (Reference Dependencies): {scores['fcn_correct_with_reference_provided_pct']:.1f}%")
print(f"Function Correctness (LLM Chain Dependencies): {scores['fcn_correct_with_llm_chain_pct']:.1f}%") 
print(f"Average Test Pass Rate: {scores['avg_tests_passed_on_reference_pct']:.1f}%")
print(f"Average Expected Failure Detection Rate: {scores['avg_expected_failures_detected_pct']:.1f}%")
```

### Understanding the Metrics

The pipeline computes four key performance dimensions:

1. **Function Correctness (Reference Dependencies)**: Percentage of tasks where the LLM's function implementation passes all test cases when using reference implementations for dependencies.

2. **Function Correctness (LLM Chain Dependencies)**: Percentage of tasks where the LLM's function implementation passes when using other LLM implementations for dependencies. This tests composition ability.

3. **Average Test Pass Rate**: For each task, computes (tests passed / total tests), then averages across all tasks. Measures test quality against reference implementations.

4. **Average Expected Failure Detection Rate**: For each task, computes (expected failures detected / total expected failures), then averages. Measures whether LLM tests can catch buggy implementations.

### Advanced Usage

#### Working with Specific Task Subsets

```python
# Load tasks then filter
pipeline.load_tasks()
basic_tasks = [t for t in pipeline.tasks if t.subcategory == "basic"]
print(f"Found {len(basic_tasks)} basic tasks")
```

#### Error Handling

```python
try:
    pipeline.load_tasks()
    pipeline.load_environment()
    pipeline.load_llm_outputs()
    results = pipeline.evaluate_all_tasks()
    scores = pipeline.compute_aggregate_score()
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Evaluation failed: {e}")
```

#### Accessing Raw Data

```python
# Access loaded data directly
print(f"Tasks: {[t.task_id for t in pipeline.tasks]}")
print(f"Environment: {pipeline.environment.environment_name}")
print(f"LLM outputs: {list(pipeline.llm_outputs.keys())}")

# Access detailed evaluation results
for result in pipeline.evaluation_results:
    if result.get("error"):
        print(f"Task {result['task_id']} failed: {result['error']}")
```

### File Organization

Your directory structure should look like:

```
your_project/
├── tasks/
│   ├── tier1/
│   │   ├── T1_SF_001.yaml
│   │   ├── T1_SF_002.yaml
│   │   └── ...
│   └── tier2/
├── environments/
│   ├── tier1_environment.yaml
│   └── tier2_environment.yaml
├── llm_outputs/
│   ├── T1_SF_001.json
│   ├── T1_SF_002.json
│   └── ...
└── results/
    ├── prompts/
    │   ├── T1_SF_001_prompt.txt
    │   └── ...
    └── individual_results/
        ├── T1_SF_001_result.json
        └── ...
```

## Key Directories

- **`tasks/`**: YAML files defining each benchmark task, organized by difficulty tier
- **`src/fem_bench/`**: Core benchmark framework code
- **`reference_solutions/`**: Verified correct implementations used for validation
- **`submissions/`**: Example solutions and user submissions for testing
- **`scripts/`**: Command-line tools for running evaluations and generating reports

### Getting Started

1. Browse tasks in `tasks/tier1/` to see available problems
2. Use `FEMBenchPipeline` to generate prompts for your LLM
3. Save LLM responses as JSON files in the expected format
4. Use the pipeline to evaluate performance and generate reports

### Resource for contributors

_template.yaml