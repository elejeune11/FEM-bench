import numpy as np
import pytest


def linear_solve(P_global, K_global, fixed, free):
    """
    Solves the linear system for displacements and internal nodal forces in a 3D linear elastic structure,
    using a partitioned approach based on fixed and free degrees of freedom (DOFs).

    The function solves for displacements at the free DOFs by inverting the corresponding submatrix
    of the global stiffness matrix (`K_ff`). A condition number check (`cond(K_ff) < 1e16`) is used
    to ensure numerical stability. If the matrix is well-conditioned, the system is solved and a nodal
    reaction vector is computed at the fixed DOFs.

    Parameters
    ----------
    P_global : ndarray of shape (n_dof,)
        The global load vector.

    K_global : ndarray of shape (n_dof, n_dof)
        The global stiffness matrix.

    fixed : array-like of int
        Indices of fixed degrees of freedom.

    free : array-like of int
        Indices of free degrees of freedom.

    Returns
    -------
    u : ndarray of shape (n_dof,)
        Displacement vector. Displacements are computed only for free DOFs; fixed DOFs are set to zero.

    nodal_reaction_vector : ndarray of shape (n_dof,)
        Nodal reaction vector. Reactions are computed only for fixed DOFs.

    Raises
    ------
    ValueError
        If the submatrix `K_ff` is ill-conditioned and the system cannot be reliably solved.
    """
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


def linear_solve_no_partition(P_global, K_global, fixed, free):
    """
    Incorrect: Ignores DOF partitioning but keeps condition number check.

    This version always solves the full system K u = P,
    even when fixed DOFs should be constrained.
    """
    condition_number = np.linalg.cond(K_global)

    if condition_number < 1e16:
        u = np.linalg.solve(K_global, P_global)
        nodal_reaction_vector = K_global @ u  # reactions mixed with forces
    else:
        raise ValueError(f"Cannot solve system: stiffness matrix is ill-conditioned (cond={condition_number:.2e})")
    return u, nodal_reaction_vector


def linear_solve_no_error_check(P_global, K_global, fixed, free):
    """
    Solves the linear system for displacements and nodal reaction forces in a 3D linear elastic structure,
    using a partitioned approach based on fixed and free degrees of freedom (DOFs), without checking
    for ill-conditioning or numerical stability.
    """
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
    Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.

    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations.
    """
    test_cases = []

    # Case 1: 2 DOFs, one fixed
    K1 = np.array([[2, 1],
                   [1, 3]], dtype=float)
    P1 = np.array([0, 4], dtype=float)
    fixed1 = [0]
    free1 = [1]
    test_cases.append((K1, P1, fixed1, free1))

    # Case 2: 3 DOFs, two free
    K2 = np.array([[4, 1, 0],
                   [1, 3, 0],
                   [0, 0, 2]], dtype=float)
    P2 = np.array([0, 9, 4], dtype=float)
    fixed2 = [0]
    free2 = [1, 2]
    test_cases.append((K2, P2, fixed2, free2))

    # Case 3: 4 DOFs, half fixed
    K3 = np.array([[10, 2, 0, 0],
                   [2, 5, 0, 0],
                   [0, 0, 3, 1],
                   [0, 0, 1, 2]], dtype=float)
    P3 = np.array([0, 0, 6, 3], dtype=float)
    fixed3 = [0, 1]
    free3 = [2, 3]
    test_cases.append((K3, P3, fixed3, free3))

    for i, (K_global, P_global, fixed, free) in enumerate(test_cases, start=1):
        u, nodal_reaction_vector = fcn(P_global, K_global, fixed, free)
        n_dof = len(P_global)

        # Check shape
        assert u.shape == (n_dof,), f"[Case {i}] Incorrect shape for u"
        assert nodal_reaction_vector.shape == (n_dof,), f"[Case {i}] Incorrect shape for nodal_reaction_vector"

        # Fixed DOFs should have zero displacement
        assert np.allclose(u[fixed], 0), f"[Case {i}] Displacement at fixed DOFs not zero"

        # Free DOFs satisfy K_ff u_f = P_f
        K_ff = K_global[np.ix_(free, free)]
        u_f = u[free]
        P_f = P_global[free]
        assert np.allclose(K_ff @ u_f, P_f), f"[Case {i}] Free DOFs do not satisfy K_ff u_f = P_f"

        # Reactions at fixed DOFs should match K_sf @ u_f - P_fixed
        K_sf = K_global[np.ix_(fixed, free)]
        expected_r_fixed = K_sf @ u_f - P_global[fixed]
        assert np.allclose(nodal_reaction_vector[fixed], expected_r_fixed), f"[Case {i}] Incorrect reactions at fixed DOFs"

        # Global equilibrium check: K u = P + reactions
        residual = K_global @ u - (P_global + nodal_reaction_vector)
        assert np.allclose(residual, 0), f"[Case {i}] Global equilibrium not satisfied"


def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """
    Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.

    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError.
    """
    # Create a nearly singular 2x2 submatrix K_ff
    # The rows are almost linearly dependent
    K_global = np.array([[1, 1, 0],
                         [1, 1 + 1e-16, 0],
                         [0, 0, 10]], dtype=float)
    
    P_global = np.array([1, 2, 3], dtype=float)
    
    # Fix the third DOF, leaving DOFs 0 and 1 free
    fixed = [2]
    free = [0, 1]  # This selects the ill-conditioned 2x2 block as K_ff

    cond = np.linalg.cond(K_global[np.ix_(free, free)])
    assert cond > 1e16, "Test setup error: K_ff is not ill-conditioned as expected"

    with pytest.raises(ValueError, match="stiffness matrix is ill-conditioned"):
        _ = fcn(P_global, K_global, fixed, free)


def task_info():
    task_id = "linear_solve"
    task_short_description = "performs a linear solve given global stiffness matrix, load vector, and free and fixed degrees of freedom"
    created_date = "2025-08-12"
    created_by = "elejeune11"
    main_fcn = linear_solve
    required_imports = ["import numpy as np", "import pytest"]
    fcn_dependencies = []
    reference_verification_inputs = [
        [
            np.array([0.0, 4.0]),
            np.array([[2.0, 1.0],
                      [1.0, 3.0]]),
            [0],
            [1]
        ],
        [
            np.array([1.0, 2.0]),
            np.array([[4.0, -1.0],
                      [-1.0, 2.0]]),
            [],
            [0, 1]
        ],
        [
            np.array([0.0, 3.0, 4.0]),
            np.array([[2.0, 0.5, 0.0],
                      [0.5, 3.0, 1.0],
                      [0.0, 1.0, 2.0]]),
            [0],
            [1, 2]
        ],
        [
            np.array([1.0, 0.0, 2.0, 0.0]),
            np.array([[10.0, 2.0, 0.0, 0.0],
                      [2.0, 5.0, 0.0, 0.0],
                      [0.0, 0.0, 3.0, 1.0],
                      [0.0, 0.0, 1.0, 2.0]]),
            [0, 2],
            [1, 3]
        ],
        [
            np.array([0.0, 0.0, 0.0, 100.0, 200.0, 300.0]),
            np.eye(6) * 100,
            [0, 1, 2],
            [3, 4, 5]
        ]
    ]
    test_cases = [{"test_code": test_linear_solve_arbitrary_solvable_cases, "expected_failures": [linear_solve_no_partition]}, {"test_code": test_linear_solve_raises_on_ill_conditioned_matrix, "expected_failures": [linear_solve_no_error_check]}]
    return task_id, task_short_description, created_date, created_by, main_fcn, required_imports, fcn_dependencies, reference_verification_inputs, test_cases
