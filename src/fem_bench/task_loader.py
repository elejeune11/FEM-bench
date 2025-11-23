from fem_bench.task_base import Task
import inspect
import textwrap


def load_task_from_info(task_info_fn) -> Task:
    info = task_info_fn()

    # Convert main function
    main_fcn_code = textwrap.dedent(inspect.getsource(info["main_fcn"]))

    # Convert dependencies
    fcn_dependency_code = [
        textwrap.dedent(inspect.getsource(dep)) for dep in info.get("fcn_dependencies", [])
    ]

    # Convert test cases
    test_cases = []
    for case in info.get("test_cases", []):
        test_code_str = textwrap.dedent(inspect.getsource(case["test_code"]))
        failure_sources = [
            textwrap.dedent(inspect.getsource(f)) for f in case.get("expected_failures", [])
        ]
        test_cases.append({
            "test_code": test_code_str,
            "expected_failures": failure_sources
        })

    return Task(
        task_id=info["task_id"],
        task_short_description=info["task_short_description"],
        created_date=info["created_date"],
        created_by=info["created_by"],
        main_fcn_code=main_fcn_code,
        required_imports=info.get("required_imports"),
        fcn_dependency_code=fcn_dependency_code,
        reference_verification_inputs=info.get("reference_verification_inputs"),
        test_cases=test_cases,
        python_version=info.get("python_version"),
        package_versions=info.get("package_versions"),
    )
