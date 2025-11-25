def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """Comprehensive test for local_geometric_stiffness_matrix_3D_beam"""
    L = 2.0
    A = 0.01
    I_rho = 0.0001
    K_g = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert K_g.shape == (12, 12)
    assert_array_almost_equal(K_g, np.zeros((12, 12)))
    K_g = fcn(L, A, I_rho, 1000.0, 100.0, 50.0, -50.0, 75.0, -75.0)
    assert_array_almost_equal(K_g, K_g.T)
    K_g_tension = fcn(L, A, I_rho, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    eigvals_tension = np.linalg.eigvals(K_g_tension)
    assert np.all(eigvals_tension >= -1e-10)
    K_g_compression = fcn(L, A, I_rho, -1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    eigvals_compression = np.linalg.eigvals(K_g_compression)
    assert np.any(eigvals_compression < 0)
    K_g_base = fcn(L, A, I_rho, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_g_moment = fcn(L, A, I_rho, 1000.0, 100.0, 50.0, 50.0, 75.0, 75.0)
    assert not np.allclose(K_g_base, K_g_moment)
    K_g_1 = fcn(L, A, I_rho, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_g_2 = fcn(L, A, I_rho, 2000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert_array_almost_equal(2 * K_g_1, K_g_2)

def test_euler_buckling_cantilever_column(fcn):
    """Test against analytical Euler buckling load for cantilever column"""
    L = 3.0
    E = 200000000000.0
    I = 1e-06
    A = 0.01
    I_rho = 2e-06
    P_cr_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
    K_e = np.zeros((12, 12))
    EI_L3 = E * I / L ** 3
    K_e[1, 1] = K_e[7, 7] = 12 * EI_L3
    K_e[1, 7] = K_e[7, 1] = -12 * EI_L3
    K_e[5, 5] = K_e[11, 11] = 4 * E * I / L
    K_e[5, 11] = K_e[11, 5] = 2 * E * I / L
    P_test = np.linspace(-1.1 * P_cr_analytical, 0, 100)
    min_eigval = np.inf
    P_cr_numerical = 0
    for P in P_test:
        K_g = fcn(L, A, I_rho, P, 0.0, 0.0, 0.0, 0.0, 0.0)
        eigvals = eigh(K_e + K_g, eigvals_only=True)
        if np.min(eigvals) < min_eigval:
            min_eigval = np.min(eigvals)
            P_cr_numerical = P
    error = abs(P_cr_numerical - P_cr_analytical) / P_cr_analytical
    assert error < 0.1