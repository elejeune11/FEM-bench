def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    diag_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    K_e_global = np.diag(diag_values)
    K_g_global = -np.eye(n_dofs)
    boundary_conditions = {}
    (lambda_crit, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    expected_lambda = np.min(diag_values)
    assert np.isclose(lambda_crit, expected_lambda, rtol=1e-10), f'Expected lambda {expected_lambda}, got {lambda_crit}'
    assert mode_shape.shape == (n_dofs,), f'Expected shape {(n_dofs,)}, got {mode_shape.shape}'
    idx_min = np.argmin(diag_values)
    assert np.abs(mode_shape[idx_min]) > 1e-10, 'Mode should have significant component at minimum eigenvalue DOF'

def test_eigen_singular_detected(fcn):
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
    K_e_global = np.diag(np.linspace(1.0, 10.0, n_dofs))
    K_e_global[0, 1] = 1e-05
    K_e_global[1, 0] = -1e-05
    K_g_global = -np.eye(n_dofs)
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
    n_nodes = 3
    n_dofs = 6 * n_nodes
    np.random.seed(42)
    A = np.random.randn(n_dofs, n_dofs)
    K_e_global = A @ A.T + np.eye(n_dofs) * 10.0
    B = np.random.randn(n_dofs, n_dofs)
    K_g_global = -(B @ B.T + np.eye(n_dofs))
    boundary_conditions = {0: np.array([True, True, True, True, True, True]), 1: np.array([True, False, False, False, False, False])}
    (lambda_1, mode_1) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    scale_factor = 2.5
    (lambda_2, mode_2) = fcn(K_e_global, scale_factor * K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(lambda_2, lambda_1 / scale_factor, rtol=1e-08), f'Expected lambda_2 = lambda_1 / {scale_factor}, but got {lambda_2} vs {lambda_1 / scale_factor}'
    assert mode_1.shape == (n_dofs,), f'Expected mode shape {(n_dofs,)}, got {mode_1.shape}'
    assert mode_2.shape == (n_dofs,), f'Expected mode shape {(n_dofs,)}, got {mode_2.shape}'
    for (node_id, constraints) in boundary_conditions.items():
        dof_start = 6 * node_id
        dof_end = dof_start + 6
        constrained_dofs = np.where(constraints)[0]
        for dof_offset in constrained_dofs:
            dof_idx = dof_start + dof_offset
            assert np.abs(mode_1[dof_idx]) < 1e-14, f'Constrained DOF {dof_idx} should be zero in mode_1'
            assert np.abs(mode_2[dof_idx]) < 1e-14, f'Constrained DOF {dof_idx} should be zero in mode_2'