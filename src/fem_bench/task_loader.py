from fem_bench.task_base import Task
import inspect
import textwrap


def load_task_from_info(task_info_fn) -> Task:
    (
        task_id,
        task_short_description,
        created_date,
        created_by,
        main_fcn,
        required_imports,
        fcn_dependencies,
        reference_verification_inputs,
        test_cases_raw,
    ) = task_info_fn()

    # Convert main function
    main_fcn_code = textwrap.dedent(inspect.getsource(main_fcn))

    # Convert dependencies
    fcn_dependency_code = [
        textwrap.dedent(inspect.getsource(dep)) for dep in fcn_dependencies
    ]

    # Convert test cases
    test_cases = []
    for case in test_cases_raw:
        test_code_str = textwrap.dedent(inspect.getsource(case["test_code"]))
        failure_sources = [
            textwrap.dedent(inspect.getsource(f)) for f in case.get("expected_failures", [])
        ]
        test_cases.append({
            "test_code": test_code_str,
            "expected_failures": failure_sources
        })

    return Task(
        task_id=task_id,
        task_short_description=task_short_description,
        created_date=created_date,
        created_by=created_by,
        main_fcn_code=main_fcn_code,
        required_imports=required_imports,
        fcn_dependency_code=fcn_dependency_code,
        reference_verification_inputs=reference_verification_inputs,
        test_cases=test_cases
    )

