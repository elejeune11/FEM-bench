def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_diag = np.array([4.0, 2.0, 3.0, 1.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_e_global = np.diag(K_e_diag)
    K_g_global = -np.eye(n_dofs)
    boundary_conditions = {}
    expected_lambda = 1.5
    expected_mode_shape = np.zeros(n_dofs)
    expected_mode_shape[3] = 1.0
    (result_lambda, result_mode) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(result_lambda, expected_lambda), 'Eigenvalue does not match expected value.'
    assert np.allclose(result_mode, expected_mode_shape), 'Mode shape does not match expected vector.'
    assert result_mode.shape == (n_dofs,), 'Mode shape vector has incorrect shape.'

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.zeros((n_dofs, n_dofs))
    K_g_global = -np.eye(n_dofs)
    boundary_conditions = {}
    with pytest.raises(ValueError, match='singular'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """
    n_nodes = 1
    n_dofs = 6
    K_e_global = np.array([[0, -1], [1, 0]])
    K_g_global = np.eye(2)
    boundary_conditions = {i: [False] * 6 for i in range(n_nodes)}
    boundary_conditions[0] = [True] * 4 + [False] * 2
    with pytest.raises(ValueError, match='complex'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 1
    n_dofs = 6
    K_e_global = np.eye(n_dofs)
    K_g_global = np.eye(n_dofs)
    boundary_conditions = {}
    with pytest.raises(ValueError, match='positive'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    np.random.seed(0)
    K_e_global = np.random.rand(n_dofs, n_dofs)
    K_e_global = K_e_global.T @ K_e_global
    K_g_base = np.random.rand(n_dofs, n_dofs)
    K_g_base = K_g_base.T @ K_g_base
    K_g_global = -K_g_base
    boundary_conditions = {0: [True] * 6}
    c = 2.5
    K_g_scaled = c * K_g_global
    (lambda_original, mode_original) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    (lambda_scaled, mode_scaled) = fcn(K_e_global, K_g_scaled, boundary_conditions, n_nodes)
    assert np.isclose(lambda_original, c * lambda_scaled), 'Eigenvalue did not scale correctly.'
    assert mode_original.shape == mode_scaled.shape == (n_dofs,), 'Mode shape vectors have incorrect shape.'
    assert np.isclose(mode_original[6:], mode_scaled[6:]).all(), 'Free DOF components of mode shapes do not match.'
    assert mode_original[:6].tolist() == [0] * 6 and mode_scaled[:6].tolist() == [0] * 6, 'Constrained DOFs are not zero.'