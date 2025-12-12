def test_eigen_known_answer(fcn):
    n_nodes = 2
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_global = -np.eye(12)
    boundary_conditions = {}
    (elastic_critical_load_factor, deformed_shape_vector) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(elastic_critical_load_factor, 1.0)
    assert deformed_shape_vector.shape == (12,)
    assert np.allclose(deformed_shape_vector, np.array([1.0] + [0.0] * 11))

def test_eigen_singluar_detected(fcn):
    n_nodes = 2
    K_e_global = np.zeros((12, 12))
    K_g_global = -np.eye(12)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    n_nodes = 1
    K_e_global = np.eye(6)
    K_g_global = np.array([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0]], dtype=float)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    n_nodes = 1
    K_e_global = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    K_g_global = -np.eye(6)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    n_nodes = 1
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    K_g_global_base = -np.eye(6)
    c = 5.0
    K_g_global_scaled = c * K_g_global_base
    (lambda1, mode1) = fcn(K_e_global, K_g_global_base, {}, n_nodes)
    (lambda2, mode2) = fcn(K_e_global, K_g_global_scaled, {}, n_nodes)
    assert np.isclose(lambda2, lambda1 / c)
    assert mode1.shape == (6,)
    assert mode2.shape == (6,)
    assert np.allclose(mode1, mode2)