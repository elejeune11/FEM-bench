def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    K_e_global = np.diag([1, 2, 3, 4, 5, 6])
    K_g_global = -np.eye(6)
    boundary_conditions = type('dummy', (object,), {'constrained_dofs': []})()
    n_nodes = 1
    (elastic_critical_load_factor, deformed_shape_vector) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    np.testing.assert_allclose(elastic_critical_load_factor, 1)
    np.testing.assert_allclose(deformed_shape_vector, np.array([1, 0, 0, 0, 0, 0]))

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    K_e_global = np.eye(6)
    K_e_global[0, 0] = 1e-17
    K_g_global = -np.eye(6)
    boundary_conditions = type('dummy', (object,), {'constrained_dofs': []})()
    n_nodes = 1
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    K_e_global = np.eye(6)
    K_g_global = np.array([[0, 1j], [-1j, 0]])
    K_g_global = np.pad(K_g_global, ((0, 4), (0, 4)), 'constant')
    boundary_conditions = type('dummy', (object,), {'constrained_dofs': [2, 3, 4, 5]})
    n_nodes = 1
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    K_e_global = -np.eye(6)
    K_g_global = np.eye(6)
    boundary_conditions = type('dummy', (object,), {'constrained_dofs': []})()
    n_nodes = 1
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    K_e_global = np.eye(6)
    K_g_global = -np.eye(6)
    c = 2.0
    boundary_conditions = type('dummy', (object,), {'constrained_dofs': []})()
    n_nodes = 1
    (elastic_critical_load_factor_1, deformed_shape_vector_1) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    (elastic_critical_load_factor_2, deformed_shape_vector_2) = fcn(K_e_global, c * K_g_global, boundary_conditions, n_nodes)
    np.testing.assert_allclose(elastic_critical_load_factor_2, elastic_critical_load_factor_1 / c)
    np.testing.assert_allclose(deformed_shape_vector_1, deformed_shape_vector_2)