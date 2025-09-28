def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 1
    K_e_global = np.diag([1, 2, 3, 4, 5, 6])
    K_g_global = -np.eye(6)
    boundary_conditions = None
    (elastic_critical_load_factor, deformed_shape_vector) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert elastic_critical_load_factor == 1
    assert np.allclose(deformed_shape_vector, np.array([1, 0, 0, 0, 0, 0]))

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned.
    """
    n_nodes = 1
    K_e_global = np.zeros((6, 6))
    K_g_global = -np.eye(6)
    boundary_conditions = None
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """
    n_nodes = 1
    K_e_global = np.diag([1, 2, 3, 4, 5, 6])
    K_g_global = np.eye(6) * 1j
    boundary_conditions = None
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 1
    K_e_global = -np.eye(6)
    K_g_global = -np.eye(6)
    boundary_conditions = None
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """
    Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size.
    """
    n_nodes = 1
    K_e_global = np.diag([1, 2, 3, 4, 5, 6])
    K_g_global = -np.eye(6)
    boundary_conditions = None
    (elastic_critical_load_factor, deformed_shape_vector) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    scale_factor = 2
    K_g_global_scaled = K_g_global * scale_factor
    (scaled_critical_load_factor, _) = fcn(K_e_global, K_g_global_scaled, boundary_conditions, n_nodes)
    assert np.isclose(scaled_critical_load_factor, elastic_critical_load_factor / scale_factor)
    assert deformed_shape_vector.shape == (6 * n_nodes,)