def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_diag = np.array([4.0, 3.0, 2.0, 1.0] + [5.0] * (n_dofs - 4))
    K_e_global = np.diag(K_e_diag)
    K_g_global = -np.eye(n_dofs)
    boundary_conditions = {0: [True] * 6, 1: [True] * 5 + [False]}
    expected_lambda = 1.0
    expected_mode_shape = np.zeros(n_dofs)
    expected_mode_shape[-1] = 1.0
    (lambda_cr, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(lambda_cr, expected_lambda), f'Expected {expected_lambda}, got {lambda_cr}'
    assert np.allclose(mode_shape, expected_mode_shape), f'Mode shape mismatch.'

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned.
    """
    n_nodes = 1
    n_dofs = 6 * n_nodes
    K_e_global = np.zeros((n_dofs, n_dofs))
    K_g_global = -np.eye(n_dofs)
    boundary_conditions = {0: [False] * 6}
    with pytest.raises(ValueError, match='singular'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """
    n_nodes = 1
    n_dofs = 6 * n_nodes
    K_e_global = np.array([[0, -1], [1, 0]])
    K_g_global = np.eye(2)
    boundary_conditions = {}
    with pytest.raises(ValueError, match='complex'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 1
    n_dofs = 6 * n_nodes
    K_e_global = -np.eye(n_dofs)
    K_g_global = np.eye(n_dofs)
    boundary_conditions = {0: [False] * 3 + [True] * 3}
    with pytest.raises(ValueError, match='positive'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size.
    """
    n_nodes = 1
    n_dofs = 6 * n_nodes
    K_e_global = np.eye(n_dofs)
    K_g_global = -np.eye(n_dofs)
    boundary_conditions = {0: [True] * 5 + [False]}
    c = 2.5
    (lambda_orig, mode_orig) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    (lambda_scaled, mode_scaled) = fcn(K_e_global, c * K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(lambda_scaled, lambda_orig / c), f'Scaling not correct: {lambda_scaled} != {lambda_orig / c}'
    assert mode_scaled.shape == mode_orig.shape, 'Mode shape vector size mismatch'
    assert mode_scaled[-1] != 0 or np.allclose(mode_scaled, mode_orig), 'Mode shape structure changed unexpectedly'