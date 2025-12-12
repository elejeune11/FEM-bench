def test_linear_solve_arbitrary_solvable_cases(fcn):
    n_nodes = 2
    P_global = np.zeros(12)
    P_global[0] = 10.0
    K_global = np.zeros((12, 12))
    K_global[0, 0] = 100.0
    K_global[1, 1] = 100.0
    K_global[2, 2] = 100.0
    K_global[3, 3] = 10.0
    K_global[4, 4] = 10.0
    K_global[5, 5] = 10.0
    K_global[6, 6] = 100.0
    K_global[7, 7] = 100.0
    K_global[8, 8] = 100.0
    K_global[9, 9] = 10.0
    K_global[10, 10] = 10.0
    K_global[11, 11] = 10.0
    K_global[0, 6] = -100.0
    K_global[6, 0] = -100.0
    K_global[1, 7] = -100.0
    K_global[7, 1] = -100.0
    K_global[2, 8] = -100.0
    K_global[8, 2] = -100.0
    K_global[3, 9] = -10.0
    K_global[9, 3] = -10.0
    K_global[4, 10] = -10.0
    K_global[10, 4] = -10.0
    K_global[5, 11] = -10.0
    K_global[11, 5] = -10.0
    boundary_conditions = {1: np.array([True, True, True, True, True, True])}
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    assert np.allclose(u[6:], 0.0)
    assert np.allclose(u[:6], [0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(r[6:], 0.0)
    assert np.allclose(r[:6], [-10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(K_global @ u, P_global + r)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    n_nodes = 2
    P_global = np.zeros(12)
    K_global = np.zeros((12, 12))
    K_global[0, 0] = 1e-16
    K_global[1, 1] = 1e-16
    K_global[2, 2] = 1e-16
    K_global[3, 3] = 1e-16
    K_global[4, 4] = 1e-16
    K_global[5, 5] = 1e-16
    K_global[6, 6] = 1.0
    K_global[7, 7] = 1.0
    K_global[8, 8] = 1.0
    K_global[9, 9] = 1.0
    K_global[10, 10] = 1.0
    K_global[11, 11] = 1.0
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)