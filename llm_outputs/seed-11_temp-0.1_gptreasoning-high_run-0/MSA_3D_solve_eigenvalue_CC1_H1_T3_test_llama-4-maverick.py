def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_global = -np.eye(12)
    boundary_conditions = {}
    (eigenvalue, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(eigenvalue, 1.0)
    assert np.allclose(mode_shape[np.argmax(np.abs(mode_shape))], 1.0)

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    K_e_global = np.eye(12)
    K_e_global[0, 0] = 0.0
    K_g_global = -np.eye(12)
    boundary_conditions = {0: np.array([False, False, False, False, False, False])}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    K_e_global = np.array([[1, 1j], [-1j, 1]])
    K_g_global = -np.eye(2)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes // 2)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 2
    K_e_global = np.eye(12)
    K_g_global = np.eye(12)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 2
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_global = -np.eye(12)
    boundary_conditions = {}
    (eigenvalue1, mode_shape1) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    c = 2.0
    (eigenvalue2, mode_shape2) = fcn(K_e_global, c * K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(eigenvalue1 / eigenvalue2, c)
    assert np.allclose(mode_shape1, mode_shape2)