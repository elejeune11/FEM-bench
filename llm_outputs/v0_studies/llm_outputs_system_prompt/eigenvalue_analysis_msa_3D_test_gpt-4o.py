def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * n_nodes)
    K_g_global = -np.eye(6 * n_nodes)
    boundary_conditions = None
    expected_load_factor = 1.0
    expected_mode_shape = np.zeros(6 * n_nodes)
    expected_mode_shape[0] = 1.0
    (load_factor, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(load_factor, expected_load_factor)
    assert np.allclose(mode_shape, expected_mode_shape)

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    K_e_global = np.zeros((6 * n_nodes, 6 * n_nodes))
    K_g_global = -np.eye(6 * n_nodes)
    boundary_conditions = None
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * n_nodes)
    K_g_global = -1j * np.eye(6 * n_nodes)
    boundary_conditions = None
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 2
    K_e_global = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0] * n_nodes)
    K_g_global = -np.eye(6 * n_nodes)
    boundary_conditions = None
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 2
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * n_nodes)
    K_g_global = -np.eye(6 * n_nodes)
    boundary_conditions = None
    (load_factor, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    c = 2.0
    K_g_scaled = c * K_g_global
    (scaled_load_factor, scaled_mode_shape) = fcn(K_e_global, K_g_scaled, boundary_conditions, n_nodes)
    assert np.isclose(scaled_load_factor, load_factor / c)
    assert np.allclose(scaled_mode_shape, mode_shape)