import numpy as np
import pytest


def MSA_3D_solve_linear_CC0_H1_T3(P_global: np.ndarray, K_global: np.ndarray, boundary_conditions: dict, n_nodes: int):
    """
    Solve for nodal displacements and support reactions in a 3D linear-elastic frame
    using a partitioned stiffness approach.

    This function partitions the global equilibrium system

        K_global * u_global = P_global

    into fixed and free degree-of-freedom (DOF) subsets based on the specified
    boundary conditions. The reduced system for the free DOFs is solved directly,
    provided that the free–free stiffness submatrix (K_ff) is well-conditioned.
    Reactions at fixed supports are then computed from the recovered free displacements.

    Parameters
    ----------
    P_global : (6*n_nodes,) ndarray of float
        Global load vector containing externally applied nodal forces and moments.
        Entries follow the per-node DOF order:
        [F_x, F_y, F_z, M_x, M_y, M_z].

    K_global : (6*n_nodes, 6*n_nodes) ndarray of float
        Assembled global stiffness matrix for the structure.

    boundary_conditions : dict[int, array-like of bool]
        Dictionary mapping each node index (0-based) to a 6-element boolean array
        defining the constrained DOFs:
            True  → fixed (prescribed zero displacement/rotation)
            False → free (unknown)
        Nodes not listed are assumed fully free.

    n_nodes : int
        Total number of nodes in the structure.

    Returns
    -------
    u : (6*n_nodes,) ndarray of float
        Global displacement vector. Free DOFs contain computed displacements;
        fixed DOFs are zero.

    r : (6*n_nodes,) ndarray of float
        Global reaction vector. Nonzero values appear only at fixed DOFs and
        represent internal support reactions:
            r_fixed = K_sf @ u_free - P_fixed.

    Raises
    ------
    ValueError
        If the reduced stiffness matrix K_ff is ill-conditioned
        (cond(K_ff) ≥ 1e16), indicating a singular or unstable system.

    Notes
    -----
    - DOF ordering per node: [u_x, u_y, u_z, θ_x, θ_y, θ_z].
    - The solution uses direct inversion of K_ff via `np.linalg.solve`.
    - A condition number check ensures numerical stability
    - Reaction forces are computed assuming zero prescribed displacements
      at fixed DOFs.
    - The system must be properly constrained (no rigid-body modes)
      for the solution to be valid.
    """
    def _partition_degrees_of_freedom(boundary_conditions: dict, n_nodes: int):
        n_dof = n_nodes * 6
        fixed = []
        for n in range(n_nodes):
            flags = boundary_conditions.get(n)
            if flags is not None:
                fixed.extend([6*n + i for i, f in enumerate(flags) if f])
        fixed = np.asarray(fixed, dtype=int)
        free = np.setdiff1d(np.arange(n_dof), fixed, assume_unique=True)
        return fixed, free

    fixed, free = _partition_degrees_of_freedom(boundary_conditions, n_nodes)
    n_dof = len(fixed) + len(free)
    K_ff = K_global[np.ix_(free,  free)]
    K_sf = K_global[np.ix_(fixed, free)]
    condition_number = np.linalg.cond(K_ff)
    if condition_number < 10 ** 16:
        u_f = np.linalg.solve(K_ff, P_global[free])
        u = np.zeros(n_dof)
        u[free] = u_f
        nodal_reaction_vector = np.zeros(P_global.shape)
        nodal_reaction_vector[fixed] = K_sf @ u_f - P_global[fixed]
    else:
        raise ValueError(f"Cannot solve system: stiffness matrix is ill-conditioned (cond={condition_number:.2e})")
    return u, nodal_reaction_vector


def linear_solve_all_ones(P_global, K_global, boundary_conditions, n_nodes):
    """
    Incorrect version (for testing): always returns arrays of ones for both
    displacement and reaction vectors, regardless of the inputs.
    """
    n_dof = K_global.shape[0]
    u = np.ones(n_dof, dtype=float)
    r = np.ones(n_dof, dtype=float)
    return u, r


def linear_solve_all_zeros(P_global, K_global, boundary_conditions, n_nodes):
    """
    Incorrect version (for testing): always returns arrays of ones for both
    displacement and reaction vectors, regardless of the inputs.
    """
    n_dof = K_global.shape[0]
    u = np.zeros(n_dof, dtype=float)
    r = np.zeros(n_dof, dtype=float)
    return u, r


def linear_solve_no_error_check(P_global, K_global, boundary_conditions, n_nodes):
    """
    Solves the linear system for displacements and nodal reaction forces in a 3D linear elastic structure,
    using a partitioned approach based on fixed and free degrees of freedom (DOFs), without checking
    for ill-conditioning or numerical stability.
    """
    def _partition_degrees_of_freedom(boundary_conditions, n_nodes):
        n_dof = n_nodes * 6
        fixed = []
        for n in range(n_nodes):
            flags = boundary_conditions.get(n)
            if flags is not None:
                fixed.extend([6*n + i for i, f in enumerate(flags) if f])
        fixed = np.asarray(fixed, dtype=int)
        free = np.setdiff1d(np.arange(n_dof), fixed, assume_unique=True)
        return fixed, free

    fixed, free = _partition_degrees_of_freedom(boundary_conditions, n_nodes)
    n_dof = len(fixed) + len(free)
    K_ff = K_global[np.ix_(free, free)]
    K_sf = K_global[np.ix_(fixed, free)]

    u_f = np.linalg.solve(K_ff, P_global[free])
    u = np.zeros(n_dof)
    u[free] = u_f

    nodal_reaction_vector = np.zeros_like(P_global)
    nodal_reaction_vector[fixed] = K_sf @ u_f - P_global[fixed]

    return u, nodal_reaction_vector


def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    # Helper to compute fixed/free from BC for a single-node (6 DOFs) toy system
    def fixed_free_from_bc(bc, n_nodes):
        assert n_nodes == 1, "This test constructs single-node toy systems."
        flags = bc.get(0, [False]*6)
        fixed = np.array([i for i, f in enumerate(flags) if f], dtype=int)
        free = np.array([i for i in range(6) if i not in fixed], dtype=int)
        return fixed, free

    test_cases = []

    # Case 1: Free DOFs = [3,4,5] (pure rotations), diagonal K_ff, no coupling to fixed -> zero reactions
    K1 = np.zeros((6, 6), dtype=float)
    K1[3, 3], K1[4, 4], K1[5, 5] = 400.0, 500.0, 600.0
    P1 = np.zeros(6, dtype=float)
    P1[3], P1[4], P1[5] = 8.0, -5.0, 3.0
    bc1 = {0: [True, True, True, False, False, False]}  # fix [0,1,2], free [3,4,5]
    n1 = 1
    test_cases.append((K1, P1, bc1, n1))

    # Case 2: Free DOFs = [0,1], coupled K_ff (2x2), no coupling to fixed -> zero reactions
    K2 = np.zeros((6, 6), dtype=float)
    K2[np.ix_([0, 1], [0, 1])] = np.array([[2.0, 1.0],
                                           [1.0, 3.0]], dtype=float)
    P2 = np.zeros(6, dtype=float)
    P2[0], P2[1] = 0.0, 4.0
    bc2 = {0: [False, False, True, True, True, True]}  # free [0,1], fix [2..5]
    n2 = 1
    test_cases.append((K2, P2, bc2, n2))

    # Case 3: Free DOFs = [2,3], coupled K_ff (2x2) with nonzero K_sf to induce reactions
    K3 = np.zeros((6, 6), dtype=float)
    # Free block
    K3[np.ix_([2, 3], [2, 3])] = np.array([[3.0, 1.0],
                                           [1.0, 2.0]], dtype=float)
    # Coupling to fixed DOFs [0,1,4,5] -> create nonzero reactions
    K3[np.ix_([0, 1, 4, 5], [2, 3])] = np.array([[0.5, -0.2],
                                                 [0.1,  0.3],
                                                 [-0.4, 0.0],
                                                 [0.0,  0.7]], dtype=float)
    K3[np.ix_([2, 3], [0, 1, 4, 5])] = K3[np.ix_([0, 1, 4, 5], [2, 3])].T
    P3 = np.zeros(6, dtype=float)
    P3[2], P3[3] = 6.0, 3.0
    bc3 = {0: [True, True, False, False, True, True]}  # fix [0,1,4,5], free [2,3]
    n3 = 1
    test_cases.append((K3, P3, bc3, n3))

    for i, (K_global, P_global, bc, n_nodes) in enumerate(test_cases, start=1):
        u, r = fcn(P_global, K_global, bc, n_nodes)
        n_dof = 6 * n_nodes

        # Basic shapes
        assert u.shape == (n_dof,), f"[Case {i}] Incorrect shape for u"
        assert r.shape == (n_dof,), f"[Case {i}] Incorrect shape for r"

        fixed, free = fixed_free_from_bc(bc, n_nodes)

        # Fixed DOFs → zero displacement
        if fixed.size:
            assert np.allclose(u[fixed], 0.0), f"[Case {i}] Fixed DOFs not zero"

        # Free DOFs satisfy K_ff u_f = P_f
        if free.size:
            K_ff = K_global[np.ix_(free, free)]
            u_f = u[free]
            P_f = P_global[free]
            assert np.allclose(K_ff @ u_f, P_f), f"[Case {i}] K_ff u_f != P_f"

        # Reactions at fixed DOFs match K_sf u_f - P_fixed
        if fixed.size and free.size:
            K_sf = K_global[np.ix_(fixed, free)]
            expected_r_fixed = (K_sf @ u[free]) - P_global[fixed]
            assert np.allclose(r[fixed], expected_r_fixed), f"[Case {i}] Incorrect reactions at fixed DOFs"

        # Global equilibrium: K u = P + r
        residual = K_global @ u - (P_global + r)
        assert np.allclose(residual, 0.0), f"[Case {i}] Global equilibrium not satisfied"


def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 1
    ndof = 6 * n_nodes

    # All DOFs free -> K_ff == K_global
    boundary_conditions = {}  # omitted node 0 => all six DOFs free

    # K_global: embed a singular (rank-1) 2x2 block; identity elsewhere
    K_global = np.eye(ndof, dtype=float)
    K_global[0:2, 0:2] = np.array([[1.0, 1.0],
                                   [2.0, 2.0]], dtype=float)

    P_global = np.arange(1.0, ndof + 1.0)

    # Sanity: make sure the condition number exceeds the solver's threshold
    cond_val = np.linalg.cond(K_global)
    assert cond_val > 1e16, f"Setup error: cond(K_ff)={cond_val:.3e} is not > 1e16"

    with pytest.raises(ValueError, match="ill-conditioned"):
        _ = fcn(P_global, K_global, boundary_conditions, n_nodes)


def task_info():
    task_id = "MSA_3D_solve_linear_CC0_H1_T3"
    task_short_description = "performs a linear solve given global stiffness matrix, load vector, and boundary conditions"
    created_date = "2025-08-12"
    created_by = "elejeune11"
    main_fcn = MSA_3D_solve_linear_CC0_H1_T3
    required_imports = ["import numpy as np", "import pytest", "from typing import Callable"]
    fcn_dependencies = []
    reference_verification_inputs = [
        # 1) Single node with 6 DOFs, all free (no constraints)
        [
            np.arange(1.0, 7.0),                       # P_global (6,)
            np.diag([10, 12, 15, 8, 9, 11]),           # Diagonal, positive definite
            {},                                        # No boundary conditions (all free)
            1,                                         # n_nodes
        ],
        # 2) Two nodes (12 DOFs), one fully fixed
        [
            np.concatenate([np.zeros(6), np.ones(6)]), # Load only on node 1
            np.diag(np.linspace(30, 80, 12)),          # Diagonal, increasing stiffnesses
            {0: [True]*6, 1: [False]*6},               # Node 0 fixed, node 1 free
            2,
        ],
        # 3) Two nodes with mixed BCs (some constrained DOFs)
        [
            np.ones(12),
            np.diag([100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155]),
            {0: [True, True, True, False, False, False],
            1: [False, False, False, False, False, False]},
            2,
        ],
        # 4) Three nodes, partial constraints (fixed base, free top)
        [
            np.ones(18),
            np.diag(np.linspace(20, 200, 18)),         # Strongly diagonal, well-conditioned
            {0: [True]*6, 1: [False, True, False, True, False, True]},
            3,
        ],
        # 5) Two nodes, coupling between free/fixed (but diagonally dominant)
        [
            np.ones(12),
            np.diag([15]*12) + 0.1*np.eye(12, k=1) + 0.1*np.eye(12, k=-1),  # Small coupling, well-conditioned
            {0: [True]*3 + [False]*3, 1: [False]*6},
            2,
        ],
        ]
    test_cases = [{"test_code": test_linear_solve_arbitrary_solvable_cases, "expected_failures": [linear_solve_all_ones, linear_solve_all_zeros]}, {"test_code": test_linear_solve_raises_on_ill_conditioned_Kff, "expected_failures": [linear_solve_no_error_check]}]
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
