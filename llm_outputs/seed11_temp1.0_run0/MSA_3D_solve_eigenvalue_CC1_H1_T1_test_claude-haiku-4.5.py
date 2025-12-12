def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    K_g_global = -np.eye(n_dofs)
    boundary_conditions = {}
    (lambda_crit, mode) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert isinstance(lambda_crit, (float, np.floating)), 'Critical load factor should be a float'
    assert lambda_crit == pytest.approx(1.0, rel=1e-10), 'Critical load factor should be 1.0'
    assert mode.shape == (n_dofs,), f'Mode shape should be ({n_dofs},), got {mode.shape}'
    assert np.all(np.isfinite(mode)), 'Mode vector should contain finite values'

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.zeros((n_dofs, n_dofs))
    K_e_global[0, 0] = 1.0
    K_g_global = np.eye(n_dofs)
    boundary_conditions = {0: [False, True, True, True, True, True], 1: [True, True, True, True, True, True]}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.array([[2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]], dtype=float)
    K_g_global = -np.eye(n_dofs)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = 0.01 * np.eye(n_dofs)
    K_g_global = -100.0 * np.eye(n_dofs)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.diag([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    K_g_global = -np.eye(n_dofs)
    boundary_conditions = {}
    (lambda_crit_1, mode_1) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    scale_factor = 2.0
    K_g_global_scaled = scale_factor * K_g_global
    (lambda_crit_2, mode_2) = fcn(K_e_global, K_g_global_scaled, boundary_conditions, n_nodes)
    expected_lambda_2 = lambda_crit_1 / scale_factor
    assert lambda_crit_2 == pytest.approx(expected_lambda_2, rel=1e-10), f'Eigenvalue should scale by 1/{scale_factor}, got {lambda_crit_2} vs expected {expected_lambda_2}'
    assert mode_1.shape == (n_dofs,), f'Mode 1 shape should be ({n_dofs},)'
    assert mode_2.shape == (n_dofs,), f'Mode 2 shape should be ({n_dofs},)'
    assert np.all(np.isfinite(mode_1)), 'Mode 1 should contain finite values'
    assert np.all(np.isfinite(mode_2)), 'Mode 2 should contain finite values'
    assert lambda_crit_1 > 0, 'Critical load factor 1 should be positive'
    assert lambda_crit_2 > 0, 'Critical load factor 2 should be positive'