def test_eigen_known_answer(fcn):
    K_e_global = np.diag([1, 2, 3, 4, 5, 6])
    K_g_global = -np.eye(6)
    boundary_conditions = type('dummy', (object,), {'constrained_dofs': np.array([])})()
    n_nodes = 1
    (elastic_critical_load_factor, deformed_shape_vector) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(elastic_critical_load_factor, 1)
    assert np.allclose(deformed_shape_vector, np.array([1, 0, 0, 0, 0, 0]))

def test_eigen_singluar_detected(fcn):
    K_e_global = np.zeros((6, 6))
    K_g_global = np.eye(6)
    boundary_conditions = type('dummy', (object,), {'constrained_dofs': np.array([])})()
    n_nodes = 1
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    K_e_global = np.eye(6)
    K_g_global = 1j * np.eye(6)
    boundary_conditions = type('dummy', (object,), {'constrained_dofs': np.array([])})()
    n_nodes = 1
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    K_e_global = -np.eye(6)
    K_g_global = np.eye(6)
    boundary_conditions = type('dummy', (object,), {'constrained_dofs': np.array([])})()
    n_nodes = 1
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    c = 2.0
    K_e_global = np.eye(6)
    K_g_global = np.eye(6)
    K_g_global_scaled = c * K_g_global
    boundary_conditions = type('dummy', (object,), {'constrained_dofs': np.array([])})()
    n_nodes = 1
    (elastic_critical_load_factor, deformed_shape_vector) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    (elastic_critical_load_factor_scaled, deformed_shape_vector_scaled) = fcn(K_e_global, K_g_global_scaled, boundary_conditions, n_nodes)
    assert np.isclose(elastic_critical_load_factor_scaled, elastic_critical_load_factor / c)
    assert deformed_shape_vector.shape == deformed_shape_vector_scaled.shape