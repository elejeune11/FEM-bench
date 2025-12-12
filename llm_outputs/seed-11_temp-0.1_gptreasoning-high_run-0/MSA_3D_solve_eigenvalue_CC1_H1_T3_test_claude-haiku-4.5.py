def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_global = -np.eye(n_dofs)
    boundary_conditions = {}
    (lambda_cr, mode) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(lambda_cr, 1.0, rtol=1e-10), f'Expected lambda=1.0, got {lambda_cr}'
    assert np.abs(mode[0]) > 1e-10, 'Mode should have significant entry at first DOF'
    assert mode.shape == (n_dofs,), f'Expected mode shape {(n_dofs,)}, got {mode.shape}'

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.zeros((n_dofs, n_dofs))
    K_g_global = np.eye(n_dofs)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.array([[1.0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.5, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]], dtype=float)
    K_g_global = np.eye(n_dofs)
    K_g_global[0, 1] = 1e-08
    K_g_global[1, 0] = -1e-08
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.eye(n_dofs)
    K_g_global = np.eye(n_dofs)
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
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_global_base = -np.eye(n_dofs)
    boundary_conditions = {}
    (lambda_cr_1, mode_1) = fcn(K_e_global, K_g_global_base, boundary_conditions, n_nodes)
    scale_factor = 2.0
    K_g_global_scaled = scale_factor * K_g_global_base
    (lambda_cr_2, mode_2) = fcn(K_e_global, K_g_global_scaled, boundary_conditions, n_nodes)
    expected_lambda_2 = lambda_cr_1 / scale_factor
    assert np.isclose(lambda_cr_2, expected_lambda_2, rtol=1e-10), f'Expected lambda_2={expected_lambda_2}, got {lambda_cr_2}'
    assert mode_1.shape == (n_dofs,), f'Expected mode_1 shape {(n_dofs,)}, got {mode_1.shape}'
    assert mode_2.shape == (n_dofs,), f'Expected mode_2 shape {(n_dofs,)}, got {mode_2.shape}'
    assert lambda_cr_1 > 0, f'Expected positive eigenvalue, got {lambda_cr_1}'
    assert lambda_cr_2 > 0, f'Expected positive eigenvalue, got {lambda_cr_2}'