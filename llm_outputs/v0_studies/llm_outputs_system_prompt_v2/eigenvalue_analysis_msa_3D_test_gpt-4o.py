def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 1
    K_e_global = np.diag([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    K_g_global = -np.eye(6)
    boundary_conditions = None
    expected_load_factor = 2.0
    expected_mode_shape = np.array([1, 0, 0, 0, 0, 0])
    (load_factor, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(load_factor, expected_load_factor)
    assert np.allclose(mode_shape, expected_mode_shape)

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 1
    K_e_global = np.zeros((6, 6))
    K_g_global = -np.eye(6)
    boundary_conditions = None
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 1
    K_e_global = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    K_g_global = np.eye(6) * 1j
    boundary_conditions = None
    with pytest.raises(ValueError, match='complex parts'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 1
    K_e_global = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    K_g_global = -np.eye(6)
    boundary_conditions = None
    with pytest.raises(ValueError, match='no positive eigenvalue'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 1
    K_e_global = np.diag([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    K_g_global = -np.eye(6)
    boundary_conditions = None
    scale_factor = 2.0
    (load_factor, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    (scaled_load_factor, scaled_mode_shape) = fcn(K_e_global, scale_factor * K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(scaled_load_factor, load_factor / scale_factor)
    assert mode_shape.shape == scaled_mode_shape.shape