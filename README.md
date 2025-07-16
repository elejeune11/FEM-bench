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

