def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    L = 10.0
    A = 1.0
    I_rho = 1.0
    K_g_zero = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(K_g_zero, np.zeros((12, 12)))
    K_g_tension = fcn(L, A, I_rho, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_g_compression = fcn(L, A, I_rho, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert not np.allclose(K_g_tension, K_g_compression)
    K_g_moments = fcn(L, A, I_rho, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert not np.allclose(K_g_moments, K_g_zero)
    K_g_scaled = fcn(L, A, I_rho, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(K_g_scaled, 2.0 * K_g_tension)
    assert np.allclose(K_g_tension, K_g_tension.T)
    assert K_g_tension.shape == (12, 12)

def test_euler_buckling_cantilever_column(fcn):
    L = 1.0
    A = 0.01
    E = 200000000000.0
    I = 1e-06
    I_rho = 1e-06
    P_critical_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
    K_g = fcn(L, A, I_rho, -1.0, 0, 0, 0, 0, 0)
    assert not np.allclose(K_g, np.zeros((12, 12)))
    tolerance = 0.1