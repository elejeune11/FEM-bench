def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 3
    K_e_global = np.diag(np.arange(1, 6 * n_nodes + 1))
    K_g_global = -np.eye(6 * n_nodes)
    boundary_conditions = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]
    (elastic_critical_load_factor, deformed_shape_vector) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(elastic_critical_load_factor, 6.0)
    assert np.allclose(deformed_shape_vector, np.array([0.0] * 6 + [0.0] * 5 + [1.0] + [0.0] * 6))

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    K_e_global = np.zeros((6 * n_nodes, 6 * n_nodes))
    K_g_global = -np.eye(6 * n_nodes)
    boundary_conditions = [0, 1, 2]
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    K_e_global = np.array([[1, 2], [-3, 4]])
    K_g_global = -np.eye(2 * n_nodes)
    boundary_conditions = []
    with pytest.raises(ValueError):
        fcn(np.kron(np.eye(3), K_e_global), K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 2
    K_e_global = -np.eye(6 * n_nodes)
    K_g_global = -np.eye(6 * n_nodes)
    boundary_conditions = []
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 2
    K_e_global = np.diag(np.arange(1, 6 * n_nodes + 1))
    K_g_global = -np.eye(6 * n_nodes)
    boundary_conditions = [0, 1, 2]
    scaling_factor = 2.5
    (elastic_critical_load_factor_ref, deformed_shape_vector_ref) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    (elastic_critical_load_factor_scaled, deformed_shape_vector_scaled) = fcn(K_e_global, scaling_factor * K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(elastic_critical_load_factor_scaled, elastic_critical_load_factor_ref / scaling_factor)
    assert deformed_shape_vector_scaled.size == 6 * n_nodes