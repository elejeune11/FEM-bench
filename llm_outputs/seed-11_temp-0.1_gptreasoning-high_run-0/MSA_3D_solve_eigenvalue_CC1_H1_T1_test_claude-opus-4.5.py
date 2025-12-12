def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    diag_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0])
    K_e_global = np.diag(diag_values)
    K_g_global = -np.eye(n_dofs)
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    (eigenvalue, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(eigenvalue, 15.0, rtol=1e-06)
    assert mode_shape.shape == (n_dofs,)
    assert np.allclose(mode_shape[:6], 0.0)
    free_mode = mode_shape[6:]
    max_idx = np.argmax(np.abs(free_mode))
    assert max_idx == 0

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.zeros((n_dofs, n_dofs))
    K_g_global = -np.eye(n_dofs)
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.eye(n_dofs)
    K_g_global = np.zeros((n_dofs, n_dofs))
    for i in range(n_dofs - 1):
        K_g_global[i, i + 1] = 1.0
        K_g_global[i + 1, i] = -1.0
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = -np.eye(n_dofs)
    K_g_global = -np.eye(n_dofs)
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    diag_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0])
    K_e_global = np.diag(diag_values)
    K_g_global = -np.eye(n_dofs)
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    (eigenvalue_1, mode_1) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    c = 2.0
    K_g_scaled = c * K_g_global
    (eigenvalue_2, mode_2) = fcn(K_e_global, K_g_scaled, boundary_conditions, n_nodes)
    assert np.isclose(eigenvalue_2, eigenvalue_1 / c, rtol=1e-06)
    assert mode_1.shape == (n_dofs,)
    assert mode_2.shape == (n_dofs,)
    assert np.allclose(mode_1[:6], 0.0)
    assert np.allclose(mode_2[:6], 0.0)