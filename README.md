# FEM-bench

A comprehensive benchmarking system for evaluating Large Language Models (LLMs) on finite element method (FEM) tasks.

**NOTE: this is a work in progress, new tasks and comprehensive results will be posted soon**

## Overview

FEM-bench evaluates LLMs through a dual-task approach:
1. **Implementation Tasks**: Generate correct finite element functions (shape functions, numerical integration, etc.)
2. **Test Generation**: Write comprehensive pytest tests that validate mathematical properties (partition of unity, interpolation conditions, etc.)

## Setup Instructions

### Prerequisites
- Python 3.10+ (3.12 recommended)

### Installation

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd fem-bench
   
   # Create virtual environment
   python3.10 -m venv fem_bench_env
   source fem_bench_env/bin/activate  # Linux/Mac
   # fem_bench_env\Scripts\activate   # Windows
   
   # Install package
   pip install --upgrade pip
   pip install -e ".[dev]"
   
   # Required packages for LLM API clients
   pip install python-dotenv requests openai google-generativeai
   ```

2. **Verify installation:**
   ```bash
   python -c "import fem_bench; print('FEM-Bench installed successfully')"
   pytest tests/  # Run tests
   ```

## Quick Start

### Using the FEMBenchPipeline

The `FEMBenchPipeline` class provides a complete automated workflow:

```python
from fem_bench.pipeline_utils import FEMBenchPipeline

# Initialize pipeline
pipeline = FEMBenchPipeline(
    tasks_dir="path/to/tasks",
    llm_outputs_dir="path/to/llm_outputs", 
    prompts_dir="path/to/prompts",
    results_dir="path/to/results"
)

# Complete evaluation workflow
pipeline.load_all_tasks()                    # Load task definitions
pipeline.generate_and_save_task_prompts()    # Generate code prompts
pipeline.generate_and_save_test_prompts()    # Generate test prompts
pipeline.load_all_llm_outputs()             # Load LLM responses
pipeline.evaluate_all_llm_outputs()         # Evaluate implementations
pipeline.evaluate_all_llm_tests()           # Evaluate test quality
scores = pipeline.compute_aggregate_score()  # Compute final scores
pipeline.create_markdown_summary()           # Creates markdown table summary of results
```

### User-Facing Pipeline API

The table below summarizes all key methods available through the `FEMBenchPipeline` class. These are the only methods needed to run the full benchmark pipeline:

| Method | Purpose |
|--------|---------|
| `load_all_tasks()` | Load all tasks from the specified `tasks_dir` and populate the internal registry. |
| `generate_and_save_task_prompts()` | Create and save code-generation prompts (`*_code_prompt.txt`). |
| `generate_and_save_test_prompts()` | Create and save test-generation prompts (`*_test_prompt.txt`). |
| `load_all_llm_outputs(allowed_llms=None)` | Load LLM-generated Python outputs from the output directory. Supports optional LLM filtering. |
| `evaluate_all_llm_outputs()` | Evaluate each LLM-generated function against the task reference function and store match results. |
| `evaluate_all_llm_tests()` | Evaluate LLM-generated test functions by running them against both reference and intentionally incorrect implementations. |
| `compute_aggregate_score()` | Compute summary metrics for each LLM including correctness, test pass rate, and expected failure detection rate. |
| `create_markdown_summary(filename="evaluation_summary.md")` | Write a Markdown report of all results to `results_dir`. |


### Core Components

- **Task Definition**: Tasks are defined as Python modules with a `task_info()` function
- **Prompt Generation**: Creates clean prompts for both implementation and test generation
- **Evaluation System**: Compares LLM outputs against reference implementations with numerical tolerance
- **Test Quality Assessment**: Evaluates whether generated tests properly validate mathematical properties
- **Scoring**: Provides aggregate metrics across all tasks and LLMs

### Output Metrics

For each LLM, the system computes:
- `fcn_correct_pct`: Percentage of correct function implementations
- `avg_tests_passed_on_reference_pct`: Average percentage of tests passed by reference implementation
- `avg_expected_failures_detected_pct`: Average percentage of expected failures correctly identified


## Overview: Interacting with the LLM API

This repo is designed to **separately manage API access logic** from the core benchmarking logic. All model-specific API clients (e.g., OpenAI, Claude, Gemini, DeepSeek) live in `llm_api/`, allowing you to:
- Swap in different providers without touching the pipeline logic.
- Cleanly isolate API-specific formatting, error handling, and key management.
- Maintain a consistent interface for generating **code** and **test functions** from prompts.

To add a new model or update an existing one, simply modify or extend the appropriate client in `llm_api/`. This modular design ensures the benchmark remains stable even as model APIs evolve.

### API Access: Setting up `.env`

To use the LLM APIs, you must create a `.env` file at the root of your project with your own API keys:

```
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_google_gemini_key_here
CLAUDE_API_KEY=your_anthropic_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
```

These keys are **not included** in the repo for security.

Supported model names: gpt-4o, gemini-flash, gemini-pro, claude-3-5, deepseek-chat

Each client loads the appropriate key using `dotenv`. Missing keys will raise a clear error during runtime.

> You only need to include keys for the models you plan to use.

### What `run_pipeline.py` Does

The `run_pipeline.py` script automates a **full LLM benchmarking cycle**:
1. **Load tasks**: Parses a directory of YAML-based task definitions.
2. **Generate prompts**: Creates code and test prompts per task.
3. **Call LLMs**: Sends prompts to each model (OpenAI, Claude, Gemini, DeepSeek), saving completions locally.
4. **Load outputs**: Reads model responses from disk.
5. **Run evaluation**:
   - Executes each generated function and test case.
   - Compares outputs to reference solutions.
6. **Score results**: Aggregates accuracy and test coverage metrics.
7. **Generate summary**: Outputs a markdown report summarizing model performance.

> The script is designed to **skip** tasks that already have saved outputs, allowing incremental runs and efficient restarts.

You can run the full benchmark directly with:

```bash
python run_pipeline.py
```

Outputs will be saved to:
- `llm_outputs/` for model completions
- `results/` for evaluation scores and `evaluation_summary.md`



## Architecture

The system consists of several key modules:

- `task_base.py`: Core `Task` class and `CodeBlock` utilities
- `task_loader.py`: Loads task definitions from Python modules
- `task_to_prompt.py`: Generates clean LLM prompts
- `evaluate_output.py`: Evaluates function correctness and test quality
- `pipeline_utils.py`: Complete evaluation pipeline orchestration

## File Structure

```
fem-bench/
├── fem_bench/
│   ├── task_base.py           # Core task definitions
│   ├── task_loader.py         # Task loading utilities
│   ├── task_to_prompt.py      # Prompt generation
│   ├── evaluate_output.py     # Evaluation logic
│   └── pipeline_utils.py      # Pipeline orchestration
├── tasks/                     # Task definitions
├── prompts/                   # Generated prompts
├── llm_outputs/              # LLM responses
├── results/                  # Evaluation results
└── tests/                    # Test suite
```

## Contributing

1. Define new tasks as Python modules with `task_info()` functions
2. Tasks should include reference implementations, test cases, and expected failures
3. Run `pytest tests/` to verify changes
4. Follow the existing code structure and documentation patterns

## Defining Tasks for LLM Code Generation

### Quick Start

Create a new task by copying the template:

```bash
cp task_template.py tasks/my_new_task.py
```

### What Tasks Define

Each task file contains:

- **Reference implementation** (`main_fcn`) - The correct solution to synthesize
- **Helper functions** - Any dependencies the main function needs  
- **Test cases** - Functions that verify correctness with expected failures
- **Input/output examples** - Sample data for automated evaluation

All components are bundled together in a `task_info()` function that returns structured metadata.

### From Task to LLM Prompt

The pipeline automatically converts each task into prompts:

**Code Generation Prompt:**
```markdown
# Python Function Implementation Task

Write a Python function that matches the exact signature and docstring below.

## Available Helper Functions:
def calculate_area(radius: float) -> float:
    """Calculate circle area."""
    return np.pi * radius**2

## Only complete the function below:
def circle_volume(radius: float, height: float) -> float:
    """Calculate cylinder volume using the helper function."""
```

**Test Generation Prompt:** Similar structure asking the LLM to write pytest-style test functions with specific names and assert statements.

The generated code is then evaluated against your reference implementation and test cases to measure LLM performance.

## Deactivating Environment
```bash
deactivate
rm -rf fem_bench_env  # To completely remove
```

## Preliminary Results

### Function Correctness (✓ = Match)

| Task                                   | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:---------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| element_stiffness_linear_elastic_1D    | ✓            | ✓               | ✓              | ×            | ✓        |
| linear_uniform_mesh_1D                 | ✓            | ✓               | ✓              | ✓            | ✓        |
| solve_linear_elastic_1D_self_contained | ✓            | ✓               | ×              | ×            | ✓        |
| Total                                  | 3/3          | 3/3             | 2/3            | 1/3          | 3/3      |

### Reference Tests Passed (%)

| Task                                   | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:---------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| element_stiffness_linear_elastic_1D    | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| linear_uniform_mesh_1D                 | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| solve_linear_elastic_1D_self_contained | 0.0%         | –               | 50.0%          | 100.0%       | 50.0%    |
| Avg Ref Pass %                         | 66.7%        | 66.7%           | 83.3%          | 100.0%       | 83.3%    |

### Expected Failures Detected (%)

| Task                                   | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:---------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| element_stiffness_linear_elastic_1D    | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| linear_uniform_mesh_1D                 | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| solve_linear_elastic_1D_self_contained | 100.0%       | –               | 100.0%         | 100.0%       | 100.0%   |
| Avg Fail Detect %                      | 100.0%       | 66.7%           | 100.0%         | 100.0%       | 100.0%   |


## Todo list
- [ ] Create additional tasks to complete the initial benchmark set  
- [ ] Iterate on prompt strategy to improve output quality  
- [ ] Validate tasks (e.g., ensure they are well-specified and testable)  
- [ ] Improve pipeline robustness and add validation checks  
- [ ] Collate and summarize results across tasks and models  