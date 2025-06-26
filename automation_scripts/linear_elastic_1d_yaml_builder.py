from fem_bench.yaml_load import create_task_from_functions, dump_task_to_yaml
import runpy
from pathlib import Path


this_dir = Path(__file__).parent

ref_path = this_dir / "linear_elastic_1d.py"
reference_code = runpy.run_path(str(ref_path))
reference_fn = reference_code["solve_linear_elastic_1D"]


fail_path = this_dir / "linear_elastic_1d_failure_examples.py"
failure_code = runpy.run_path(str(fail_path))
fail_fn = failure_code["solve_linear_elastic_1D_always_return_zeros"]

# 1. Bundle failures into a dictionary
failure_fns = {
    "always_zero": fail_fn
}

# 2. Create the Task object
task = create_task_from_functions(
    task_id="T1_FE_00X",  # <-- your desired task ID
    reference_fn=reference_fn,
    failure_fns=failure_fns,
    title="FEM Solver",
    short_description="Simple FEA solver example.",
    created_by="elejeune11"
)

# 3. (Optional) Save to YAML file
output_path = Path(__file__).parent / "T1_FE_00X.yaml"
dump_task_to_yaml(task, output_path)

print(f"Task written to: {output_path}")

aa = 44



aa = 44