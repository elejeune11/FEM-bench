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
   
   # Additional dependencies for prompting
   pip install selenium
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
```

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

## Deactivating Environment
```bash
deactivate
rm -rf fem_bench_env  # To completely remove
```

## Preliminary Results

claude4 = claude sonnet 4

gemini = gemini 2.0 flash-lite

gpt41 = openAI GPT-4.1

maiu = My AI University app see: https://my-ai-university.com/ and https://huggingface.co/spaces/my-ai-university/finite-element-method

### Function Correctness (✓ = Match)

| Task                                   | claude4   | gemini   | gpt41   | maiu   |
|:---------------------------------------|:----------|:---------|:--------|:-------|
| element_stiffness_linear_elastic_1D    | ✓         | ✓        | ✓       | ×      |
| linear_uniform_mesh_1D                 | ✓         | ✓        | ✓       | ✓      |
| solve_linear_elastic_1D_self_contained | ×         | ×        | ✓       | ✓      |
| Total                                  | 2/3       | 2/3      | 3/3     | 2/3    |

### Reference Tests Passed (%)

| Task                                   | claude4   | gemini   | gpt41   | maiu   |
|:---------------------------------------|:----------|:---------|:--------|:-------|
| element_stiffness_linear_elastic_1D    | 0.0%      | 0.0%     | 0.0%    | 0.0%   |
| linear_uniform_mesh_1D                 | 100.0%    | 100.0%   | 100.0%  | 100.0% |
| solve_linear_elastic_1D_self_contained | 50.0%     | 50.0%    | 100.0%  | –      |
| Avg Ref Pass %                         | 50.0%     | 50.0%    | 66.7%   | 50.0%  |

### Expected Failures Detected (%)

| Task                                   | claude4   | gemini   | gpt41   | maiu   |
|:---------------------------------------|:----------|:---------|:--------|:-------|
| element_stiffness_linear_elastic_1D    | 100.0%    | 100.0%   | 100.0%  | 100.0% |
| linear_uniform_mesh_1D                 | 100.0%    | 100.0%   | 100.0%  | 100.0% |
| solve_linear_elastic_1D_self_contained | 100.0%    | 100.0%   | 100.0%  | –      |
| Avg Fail Detect %                      | 100.0%    | 100.0%   | 100.0%  | 100.0% |



## Todo list
[] Create additional tasks to form a complete initial benchmark
[] Set up automation scripts to call LLM APIs
[] Collate results