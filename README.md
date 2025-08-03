# FEM-bench

[![python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/sandialabs/sibl#license)

[![codecov](https://codecov.io/gh/elejeune11/FEM-bench/graph/badge.svg?token=p5DMvJ6byO)](https://codecov.io/gh/elejeune11/FEM-bench)
[![tests](https://github.com/elejeune11/FEM-bench/actions/workflows/test.yml/badge.svg)](https://github.com/elejeune11/FEM-bench/actions)


A comprehensive benchmarking system for evaluating Large Language Models (LLMs) on finite element method (FEM) tasks.

**NOTE: this is a work in progress, new tasks and results will be posted as they are created. If you have questions or comments, please feel free to contact me!**

## Table of Contents

* [Overview](#overview)
    - [File Structure](#files)
* [FEM-Bench Setup Instructions](#setup)
* [Reproducing Our Results](#repro)
    - [Latest Results](#results)
    - [How To Run the Pipeline](#run_pipeline)
    - [User-Facing Pipeline API](#pipeline_api)
* [Extending to Additional LLMs](#other_llms)
* [Creating New Tasks](#new_tasks)
    - [From Task to LLM Prompt](#task_to_llm)
* [Citation Info](#cite)
* [TODO List](#todo)


## Overview <a name="overview"></a>

FEM-bench evaluates LLMs through a dual-task approach:
* **Implementation tasks**: Generate correct finite element functions (shape functions, numerical integration, etc.)
* **Test generation**: Write comprehensive pytest tests that validate mathematical properties (partition of unity, interpolation conditions, etc.)

You can think of FEM-Bench as having three parts:
* **The core software**: This contains all logic for loading tasks, generating prompts, evaluating function correctness, and computing benchmark metrics. It is designed to be model-agnostic, reproducible, and modular, enabling consistent evaluation regardless of the LLM used.

* **The LLM API evaluation**: API clients for models like GPT-4, Deepseek, Claude, and Gemini are isolated in a separate module to support easy extension, cleaner testing, and secure handling of API keys. This separation ensures that model-specific logic doesn’t pollute the core benchmarking pipeline and allows offline re-evaluation using saved outputs.

* **Tasks**: Each task defines a reference implementation, test cases, and metadata for both code and test generation. These form the basis for evaluating LLM performance on well-defined FEM-related coding challenges.

**A major goal of this tool is to make it easy to create and deploy new Tasks, ensuring the system stays relevant and highly extensible.**

### File Structure <a name="files"></a>

```
fem-bench/
├── .env                       # API keys for LLM access
├── fem_bench_env/             # Virtual environment
├── LICENSE                    # License file
├── llm_api/                   # API client wrappers for LLMs
├── llm_outputs/               # LLM responses
├── prompts/                   # Generated prompts
├── pyproject.toml             # Project metadata and dependencies
├── README.md                  # Project README
├── results/                   # Evaluation results
├── run_pipeline.py            # Script to run the full benchmarking pipeline
├── src/
│   └── fem_bench/
│       ├── __init__.py
│       ├── evaluate_output.py     # Evaluation logic
│       ├── pipeline_utils.py      # Pipeline orchestration
│       ├── task_base.py           # Core task definitions
│       ├── task_loader.py         # Task loading utilities
│       ├── task_to_prompt.py      # Prompt generation
│       └── fem_bench.egg-info/    # Metadata for installed package
├── task_template.py          # Template for defining new tasks
├── tasks/                    # Task definitions
└── tests/                    # Test suite

```

## FEM-Bench Setup Instructions <a name="setup"></a>

### Prerequisites
- Python 3.10+ (3.11 and 3.12 should work, 3.10 has been tested most extensively)

### Installation

1. **Clone and setup:**
   ```bash
   git clone https://github.com/elejeune11/FEM-bench
   cd fem-bench
   
   # Create virtual environment
   python3.10 -m venv fem_bench_env
   source fem_bench_env/bin/activate  # Linux/Mac
   # fem_bench_env\Scripts\activate   # Windows
   
   # Install package
   pip install --upgrade pip
   pip install -e ".[dev]"
   
   # Required packages for LLM API clients (for setup shown in the repo)
   pip install python-dotenv requests openai google-generativeai
   ```

2. **Verify installation:**
   ```bash
   python -c "import fem_bench; print('FEM-Bench installed successfully')"
   pytest --cov=fem_bench --cov-report=term-missing -v tests/
   ```

**Deactivating Environment**

```bash
deactivate
rm -rf fem_bench_env  # To completely remove
```

## Reproducing Our Results <a name="repro"></a>

### Latest Results <a name="results"></a>

Output Metrics:
* **Function Correctness (✓ = Match)**: Indicates whether each model's generated function produced outputs that exactly matched the reference implementation on all verification inputs.
* **Reference Tests Passed (%)**: Shows the percentage of model-generated test functions that successfully passed when run against the known-correct reference implementation.
* **Expected Failures Detected (%)**: Reports how often each model's test functions correctly failed when run on known-broken implementations, assessing the tests’ ability to catch errors.

### Function Correctness (✓ = Match)

| Task                                   | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:---------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| beam_transformation_matrix_3D          | ×            | ×               | ×              | ×            | ✓        |
| element_stiffness_linear_elastic_1D    | ✓            | ✓               | ✓              | ×            | ✓        |
| linear_uniform_mesh_1D                 | ✓            | ✓               | ✓              | ✓            | ✓        |
| local_elastic_stiffness_matrix_3D_beam | ✓            | ✓               | ×              | ×            | ✓        |
| solve_linear_elastic_1D_self_contained | ✓            | ✓               | ×              | ×            | ✓        |
| Total                                  | 4/5          | 4/5             | 2/5            | 1/5          | 5/5      |

### Reference Tests Passed (%)

| Task                                   | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:---------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| beam_transformation_matrix_3D          | 66.7%        | 66.7%           | 66.7%          | 66.7%        | 66.7%    |
| element_stiffness_linear_elastic_1D    | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| linear_uniform_mesh_1D                 | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| local_elastic_stiffness_matrix_3D_beam | 0.0%         | 0.0%            | 0.0%           | 0.0%         | 100.0%   |
| solve_linear_elastic_1D_self_contained | 0.0%         | –               | 50.0%          | 100.0%       | 50.0%    |
| Avg Ref Pass %                         | 53.3%        | 53.3%           | 63.3%          | 73.3%        | 83.3%    |

### Expected Failures Detected (%)

| Task                                   | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:---------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| beam_transformation_matrix_3D          | 50.0%        | 100.0%          | 50.0%          | 75.0%        | 50.0%    |
| element_stiffness_linear_elastic_1D    | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| linear_uniform_mesh_1D                 | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| local_elastic_stiffness_matrix_3D_beam | 100.0%       | 0.0%            | 100.0%         | 100.0%       | 100.0%   |
| solve_linear_elastic_1D_self_contained | 100.0%       | –               | 100.0%         | 100.0%       | 100.0%   |
| Avg Fail Detect %                      | 90.0%        | 60.0%           | 90.0%          | 95.0%        | 90.0%    |


### How To Run the Pipeline <a name="run_pipeline"></a>

To use the LLM APIs, you must create a `.env` file at the root of your project with your own API keys:

```
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_google_gemini_key_here
CLAUDE_API_KEY=your_anthropic_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
```

These keys are **not included** in the repo for security. Each client loads the appropriate key using `dotenv`. Missing keys will raise a clear error during runtime. The current code supports: gpt-4o, gemini-flash, gemini-pro, claude-3-5, deepseek-chat. You can modify the code to change this, and you only need to include keys for the models you plan to use.

The `run_pipeline.py` script automates a **full LLM benchmarking cycle**:
1. **Load tasks** from the `tasks/` directory.
2. **Generate prompts** for function and test synthesis, saved to `prompts/`.
3. **Call LLMs** to generate code and test files for each task:
   - Outputs are saved to `llm_outputs/`
   - Skips generation if outputs already exist
4. **Load generated completions** into the evaluation pipeline.
5. **Evaluate generated functions** against reference outputs.
6. **Evaluate generated test files** against correct and intentionally broken implementations.
7. **Aggregate results** and generate a Markdown summary in `results/`.

You can run the full benchmark directly with:

```bash
python run_pipeline.py
```

Outputs will be saved to:
- `llm_outputs/` for model completions
- `results/` for evaluation scores and `evaluation_summary.md`

### User-Facing Pipeline API <a name="pipeline_api"></a>

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



## Extending to Additional LLMs <a name="other_llms"></a>

This repo is designed with modularity in mind: all model-specific API logic is isolated in the `llm_api/` folder, keeping the core benchmarking pipeline clean and stable.

To change the models used in the benchmark, update the `MODEL_NAMES` list in your pipeline script (e.g., `["gpt-4o", "claude-3-5", ...]`).

To add a new model:
- Create a new `*_client.py` file in `llm_api/` that defines `call_<model>_for_code()` and `call_<model>_for_tests()`.
- Update `llm_clients.py` to route the new `model_name` string to your client functions.

This setup makes it easy to support new providers, customize request/response handling, and preserve a unified interface for generating both function and test completions.


## Creating New Tasks <a name="new_tasks"></a>

Each task defines a single finite element related computation, including a reference implementation, test cases, and known failure modes. Tasks are modular and self-contained, enabling evaluation of both function generation and test synthesis. Examples include mesh generation, element stiffness assembly, and complete problem solvers. Before creating a new task, we recommend looking at already created tasks as examples.

To create a new task, you can start by copying the template:

```bash
cp task_template.py tasks/my_new_task.py
```

Each task file contains:

- **A reference implementation**: The correct function to be learned or reproduced by the model.
- **Helper functions (optional)**: Dependencies used by the main function, such as shape functions or numerical integrators.
- **Reference verification inputs**: A list of example inputs used to evaluate output correctness.
- **Test functions**: Pytest-style tests designed to check correctness, robustness, and behavior.
- **Expected failure cases**: One or more incorrect implementations that the test functions should catch.
- **Metadata**: Information like task ID, author, creation date, and a short description.

All components are bundled together in a `task_info()` function that returns structured metadata.

### From Task to LLM Prompt <a name="task_to_llm"></a>

Each Task defines a Python function to implement and is automatically converted into two structured prompts: one for function generation, and one for test synthesis. The code-generation prompt includes the target function’s exact signature and docstring, plus any helper functions and import restrictions. The test-generation prompt presents the same function and lists all test functions to implement, along with their names and docstrings.

To add a new Task, you only need to define its metadata, reference implementation, and test cases — the system handles formatting and prompt generation automatically. However, keep in mind that the information included in the function docstring will make it into the prompt.

#### Example Tasks:

Very simple Task:
* [linear_uniform_mesh_1D.py](tasks/linear_uniform_mesh_1D.py)

Simple Task defined with mulitple function dependencies:
* [element_stiffness_linear_elastic_1D.py](tasks/element_stiffness_linear_elastic_1D.py)

Slightly more complicated Task:
* [solve_linear_elastic_1D_self_contained](tasks/solve_linear_elastic_1D_self_contained.py)

#### Example Generated Code Prompts:
* [linear_uniform_mesh_1D_code_prompt.txt](prompts/linear_uniform_mesh_1D_code_prompt.txt)
* [element_stiffness_linear_elastic_1D_code_prompt.txt](prompts/element_stiffness_linear_elastic_1D_code_prompt.txt)
* [solve_linear_elastic_1D_self_contained_code_prompt.txt](prompts/solve_linear_elastic_1D_self_contained_code_prompt.txt)


#### Example Generated Test Prompts:
* [linear_uniform_mesh_1D_test_prompt.txt](prompts/linear_uniform_mesh_1D_test_prompt.txt)
* [element_stiffness_linear_elastic_1D_test_prompt.txt](prompts/element_stiffness_linear_elastic_1D_test_prompt.txt)
* [solve_linear_elastic_1D_self_contained_test_prompt.txt](prompts/solve_linear_elastic_1D_self_contained_test_prompt.txt)


## Citation Info <a name="cite"></a>

After we have more Tasks completed, we will prepare a manuscript on our results. For now, if you use our work please cite this GitHub page:

```
@software{lejeune2024fem_bench,
  author = {Emma Lejeune},
  title = {FEM-Bench: A Comprehensive Benchmarking System for Evaluating Large Language Models on Finite Element Method Tasks},
  url = {https://github.com/elejeune11/FEM-bench},
  version = {main},
  year = {2024},
  note = {GitHub repository},
  keywords = {finite element method, large language models, benchmarking, machine learning}
}
```


## TODO list <a name="todo"></a>
- [ ] Create additional tasks to complete the initial benchmark set  
- [ ] Iterate on prompt strategy to improve output quality (maybe)
- [ ] Investigate the role of LLM temperature on performance (maybe)
- [ ] Improve pipeline robustness i.e., what issues arrise with new tasks? 
- [ ] Continue to collate and summarize results across tasks and models