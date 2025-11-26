def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 2
    n_dof = 6 * n_nodes
    boundary_conditions = {0: np.full(6, True)}
    K_e_global = np.diag(np.arange(1, n_dof + 1, dtype=float))
    K_g_global = -np.eye(n_dof)
    expected_lambda = 7.0
    expected_mode = np.zeros(n_dof)
    expected_mode[6] = 1.0
    (lambda_crit, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(lambda_crit, expected_lambda)
    assert mode_shape.shape == (n_dof,)
    assert np.allclose(mode_shape[:6], 0.0)
    free_mode = mode_shape[6:]
    max_abs_idx = np.argmax(np.abs(free_mode))
    assert max_abs_idx == 0
    normalized_mode = free_mode / free_mode[max_abs_idx]
    assert np.allclose(normalized_mode, expected_mode[6:])

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned.
    """
    n_nodes = 2
    n_dof = 6 * n_nodes
    boundary_conditions = {0: np.full(6, True)}
    K_e_global = np.zeros((n_dof, n_dof))
    K_g_global = -np.eye(n_dof)
    with pytest.raises(ValueError, match='ill-conditioned/singular'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """
    n_nodes = 1
    n_dof = 6 * n_nodes
    boundary_conditions = {}
    K_e_global = np.zeros((n_dof, n_dof))
    K_e_global[0, 1] = 1.0
    K_e_global[1, 0] = -1.0
    K_g_global = -np.eye(n_dof)
    with pytest.raises(ValueError, match='non-negligible complex parts'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 1
    n_dof = 6 * n_nodes
    boundary_conditions = {}
    K_e_global = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    K_g_global = -np.eye(n_dof)
    with pytest.raises(ValueError, match='no positive eigenvalue'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """
    Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size.
    """
    n_nodes = 2
    n_dof = 6 * n_nodes
    boundary_conditions = {0: np.full(6, True)}
    K_e_global = np.diag(np.arange(1, n_dof + 1, dtype=float))
    K_g_global_base = -np.eye(n_dof)
    (lambda_base, mode_base) = fcn(K_e_global, K_g_global_base, boundary_conditions, n_nodes)
    scale_factor = 2.5
    K_g_global_scaled = scale_factor * K_g_global_base
    (lambda_scaled, mode_scaled) = fcn(K_e_global, K_g_global_scaled, boundary_conditions, n_nodes)
    assert np.isclose(lambda_scaled, lambda_base / scale_factor)
    assert mode_scaled.shape == (n_dof,)
    assert mode_scaled.shape == mode_base.shape
    norm_base = mode_base / np.linalg.norm(mode_base)
    norm_scaled = mode_scaled / np.linalg.norm(mode_scaled)
    assert np.allclose(np.abs(norm_base), np.abs(norm_scaled))