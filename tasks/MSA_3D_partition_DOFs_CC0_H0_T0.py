import numpy as np
import pytest


def MSA_3D_partition_DOFs_CC0_H0_T0(boundary_conditions: dict, n_nodes: int):
    """
    Partition global degrees of freedom (DOFs) into fixed and free sets for a 3D frame.

    Purpose
    -------
    For a 3D frame with 6 DOFs per node (3 translations and 3 rotations), this routine
    builds the global index sets of fixed and free DOFs used to assemble, partition,
    and solve the linear system. The global DOF ordering per node is:
        [UX, UY, UZ, RX, RY, RZ].
    Global indices are 0-based and assigned as: DOF_index = 6*node + local_dof.

    Parameters
    ----------
    boundary_conditions : dict[int, array-like of bool]
        Maps node index (0-based, in [0, n_nodes-1]) to a 6-length iterable of booleans
        indicating which DOFs are fixed at that node (True = fixed, False = free).
        Nodes not present in the dict are assumed to have all DOFs free.
    n_nodes : int
        Total number of nodes in the structure (N ≥ 0).

    Returns
    -------
    fixed : ndarray of int, shape (n_fixed,)
        Sorted, unique global indices of fixed DOFs (0-based). May be empty if no DOFs are fixed.
    free : ndarray of int, shape (6*N - n_fixed,)
        Sorted, unique global indices of free DOFs (0-based). Disjoint from `fixed`.
        The union of `fixed` and `free` covers all DOFs: {0, 1, ..., 6*N - 1}.
    """
    n_dof = n_nodes * 6
    fixed = []
    for n in range(n_nodes):
        flags = boundary_conditions.get(n)
        if flags is not None:
            fixed.extend([6*n + i for i, f in enumerate(flags) if f])
    fixed = np.asarray(fixed, dtype=int)
    free = np.setdiff1d(np.arange(n_dof), fixed, assume_unique=True)
    return fixed, free


def partition_degrees_of_freedom_all_free(boundary_conditions, n_nodes):
    """
    Incorrect version: ignores all boundary conditions and returns every DOF as free.
    """
    n_dof = n_nodes * 6
    fixed = np.array([], dtype=int)
    free = np.arange(n_dof, dtype=int)
    return fixed, free


def partition_degrees_of_freedom_all_fixed(boundary_conditions, n_nodes):
    """
    Incorrect version: marks all DOFs as fixed regardless of input.
    """
    n_dof = n_nodes * 6
    fixed = np.arange(n_dof, dtype=int)
    free = np.array([], dtype=int)
    return fixed, free


def test_partition_dofs_correctness(fcn):
    """
    Test that partition_degrees_of_freedom correctly separates fixed and free DOFs
    across representative cases including no constraints, full constraints,
    partial constraints, and nonconsecutive node constraints.
    """
    # A) One node, no constraints
    fixed, free = fcn({}, 1)
    np.testing.assert_array_equal(fixed, np.array([], dtype=int))
    np.testing.assert_array_equal(free, np.arange(6, dtype=int))

    # B) One node, all fixed
    fixed, free = fcn({0: [True]*6}, 1)
    np.testing.assert_array_equal(fixed, np.arange(6, dtype=int))
    np.testing.assert_array_equal(free, np.array([], dtype=int))

    # C) Two nodes, partial constraints
    bcs = {
        0: [True, True, True, False, False, False],  # fix first 3 DOFs
        1: [False, False, True, False, False, True],  # fix DOFs 2 and 5 of node 1
    }
    fixed, free = fcn(bcs, 2)
    expected_fixed = np.array([0, 1, 2, 8, 11], dtype=int)
    np.testing.assert_array_equal(fixed, np.sort(expected_fixed))
    np.testing.assert_array_equal(
        free, np.setdiff1d(np.arange(12, dtype=int), expected_fixed)
    )

    # D) Three nodes, nonconsecutive constraints (nodes 0 and 2)
    bcs = {
        0: [True, False, False, False, False, True],
        2: [True, False, False, True, False, False],
    }
    fixed, free = fcn(bcs, 3)
    expected_fixed = np.array([0, 5, 12, 15], dtype=int)
    np.testing.assert_array_equal(fixed, np.sort(expected_fixed))
    np.testing.assert_array_equal(
        free, np.setdiff1d(np.arange(18, dtype=int), expected_fixed)
    )


def task_info():
    task_id = "MSA_3D_partition_DOFs_CC0_H0_T0"
    task_short_description = "partitions DOFs into fixed and free for 3D MSA problems"
    created_date = "2025-11-21"
    created_by = "elejeune11"
    main_fcn = MSA_3D_partition_DOFs_CC0_H0_T0
    required_imports = ["import numpy as np", "import pytest", "from typing import Callable"]
    fcn_dependencies = []
    reference_verification_inputs = [
        # 1) Single node, no constraints
        [{}, 1],
        # 2) Single node, all fixed
        [{0: [True, True, True, True, True, True]}, 1],
        # 3) Two nodes, partial constraints on both
        [
            {
                0: [True, False, True, False, False, False],
                1: [False, False, True, False, False, True],
            },
            2,
        ],
        # 4) Ten nodes, nonconsecutive constraints
        [
            {
                0: [True, False, False, False, False, True],
                5: [True, False, False, True, False, False],
            },
            10,
        ],
        # 5) Six nodes, mixed constraint patterns
        [
            {
                0: [True, False, False, False, False, False],
                1: [False, True, False, False, False, False],
                2: [False, False, True, False, False, False],
                3: [False, False, False, True, False, False],
                4: [False, False, False, False, True, False],
                5: [False, False, False, False, False, True],
            },
            6,
        ],
    ]
    test_cases = [{"test_code": test_partition_dofs_correctness, "expected_failures": [partition_degrees_of_freedom_all_free, partition_degrees_of_freedom_all_fixed]}]
    return {
        "task_id": task_id,
        "task_short_description": task_short_description,
        "created_date": created_date,
        "created_by": created_by,
        "main_fcn": main_fcn,
        "required_imports": required_imports,
        "fcn_dependencies": fcn_dependencies,
        "reference_verification_inputs": reference_verification_inputs,
        "test_cases": test_cases,
        # "python_version": "version_number",
        # "package_versions": {"numpy": "version_number", },
    }