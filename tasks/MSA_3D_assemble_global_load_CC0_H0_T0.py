import numpy as np
import pytest


def MSA_3D_assemble_global_load_CC0_H0_T0(nodal_loads:dict, n_nodes: int):
    """
    Assemble the global nodal load vector for a 3D linear-elastic frame structure.
    Constructs the global right-hand-side vector (P) for the equilibrium equation:
        K * u = P
    where K is the assembled global stiffness matrix and u is the global displacement
    vector. Each node contributes up to six DOFs corresponding to forces and moments
    in the global Cartesian frame.

    Parameters
    ----------
    nodal_loads : dict[int, array-like of float]
        Mapping from node index (0-based) to a 6-component load vector:
            [F_x, F_y, F_z, M_x, M_y, M_z]
        representing forces (N) and moments (N·m) applied at that node in
        **global coordinates**. Nodes not listed are assumed to have zero loads.
    n_nodes : int
        Total number of nodes in the structure. Must be consistent with the
        indexing used in `nodal_loads`.
    Returns
    -------
    P : (6 * n_nodes,) ndarray of float
        Global load vector containing all nodal forces and moments.
        DOF ordering per node: [UX, UY, UZ, RX, RY, RZ].
        Entries for unconstrained or unloaded nodes are zero.
    Notes
    -----
    - DOF indices are 0-based and grouped sequentially by node:
        DOFs for node n → [6*n, 6*n+1, 6*n+2, 6*n+3, 6*n+4, 6*n+5].
    - Loads are assumed to be applied in the global coordinate system.
    - Input vectors are automatically converted to float.
    """
    n_dof = 6 * n_nodes

    def _node_dofs(n):  # 6 global DOFs for node n
        return list(range(6*n, 6*n + 6))

    P = np.zeros(n_dof)
    for n, load in nodal_loads.items():
        P[_node_dofs(n)] += np.asarray(load, dtype=float)
    return P


def MSA_3D_assemble_global_load_CC0_H0_T0_incorrect(nodal_loads: dict, n_nodes: int):
    """
    BUGGY version (for test validation): writes loads in the wrong DOF order.
    Expected per-node order: [Fx, Fy, Fz, Mx, My, Mz]
    This version incorrectly maps as: [Fy, Fz, Fx, My, Mz, Mx]
    """
    n_dof = 6 * n_nodes
    P = np.zeros(n_dof, dtype=float)

    def _node_dofs(n):
        return slice(6*n, 6*n + 6)

    for n, load in nodal_loads.items():
        L = np.asarray(load, dtype=float)
        if L.shape != (6,):
            # keep behavior simple; let the test harness catch shape issues if any
            L = L.reshape(-1)[:6] if L.size >= 6 else np.pad(L, (0, 6 - L.size))
        # WRONG reordering:
        L_wrong = np.array([L[1], L[2], L[0], L[4], L[5], L[3]], dtype=float)
        P[_node_dofs(n)] += L_wrong

    return P


def test_MSA_3D_assemble_global_load_CC0_H0_T0_comprehensive(fcn):
    """
    Comprehensive correctness test for MSA_3D_assemble_global_load_CC0_H0_T0:
    - Basic correctness (single and multiple nodes)
    - Zeros for unspecified nodes, proper global indexing
    - Randomized consistency vs manual reference assembly
    """
    # --- 1) Single node, exact mapping
    n_nodes = 1
    loads = {0: [1, 2, 3, 4, 5, 6]}
    P = fcn(loads, n_nodes)
    assert isinstance(P, np.ndarray)
    assert P.dtype == float
    np.testing.assert_array_equal(P, np.array([1, 2, 3, 4, 5, 6], dtype=float))

    # --- 2) Multiple nodes, partial specification
    n_nodes = 4
    loads = {
        0: [0.0, -10.0, 0.0, 0.0, 0.0, 1.0],
        2: [5.0, 0.0, 0.0, 0.0, -2.0, 0.0],
    }
    P = fcn(loads, n_nodes)
    assert P.shape == (6 * n_nodes,)
    # Node 0 block
    np.testing.assert_array_equal(P[0:6], np.array([0.0, -10.0, 0.0, 0.0, 0.0, 1.0]))
    # Node 1 (unspecified)
    np.testing.assert_array_equal(P[6:12], np.zeros(6))
    # Node 2 block
    np.testing.assert_array_equal(P[12:18], np.array([5.0, 0.0, 0.0, 0.0, -2.0, 0.0]))
    # Node 3 (unspecified)
    np.testing.assert_array_equal(P[18:24], np.zeros(6))
    assert np.count_nonzero(P) == 4  # only nonzero entries above

    # --- 3) Randomized consistency test
    rng = np.random.default_rng(42)
    n_nodes = 5
    loads = {0: rng.random(6), 2: rng.random(6), 4: rng.random(6)}
    P_ref = np.zeros(6 * n_nodes)
    for n, v in loads.items():
        P_ref[6 * n : 6 * n + 6] += v.astype(float)
    P = fcn(loads, n_nodes)
    np.testing.assert_allclose(P, P_ref, rtol=0, atol=0)


def task_info():
    task_id = "MSA_3D_assemble_global_load_CC0_H0_T0"
    task_short_description = "assemble global load vector for 3D MSA problems"
    created_date = "2025-11-21"
    created_by = "elejeune11"
    main_fcn = MSA_3D_assemble_global_load_CC0_H0_T0
    required_imports = ["import numpy as np", "import pytest", "from typing import Callable"]
    fcn_dependencies = []
    reference_verification_inputs = [
        # 1) Single node, simple load
        [{0: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, 1],
        # 2) Multiple nodes, partial specification
        [
            {
                0: [0.0, -10.0, 0.0, 0.0, 0.0, 1.0],
                2: [5.0, 0.0, 0.0, 0.0, -2.0, 0.0],
            },
            4,
        ],
        # 3) Ten nodes, nonconsecutive constraints
        [
            {
                0: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                5: [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            },
            10,
        ],
        # 4) Six nodes, varied loads
        [
            {
                0: [1, 0, 0, 0, 0, 0],
                1: [0, 1, 0, 0, 0, 0],
                2: [0, 0, 1, 0, 0, 0],
                3: [0, 0, 0, 1, 0, 0],
                4: [0, 0, 0, 0, 1, 0],
                5: [0, 0, 0, 0, 0, 1],
            },
            6,
        ],
    ]
    test_cases = [{"test_code": test_MSA_3D_assemble_global_load_CC0_H0_T0_comprehensive, "expected_failures": [MSA_3D_assemble_global_load_CC0_H0_T0_incorrect]}]
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