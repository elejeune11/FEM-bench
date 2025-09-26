def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """Comprehensive test for local_geometric_stiffness_matrix_3D_beam."""
    import numpy as np
    (L, A, I_rho) = (2.0, 0.01, 0.0001)
    K_g_zero = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert K_g_zero.shape == (12, 12)
    assert np.allclose(K_g_zero, np.zeros((12, 12)))
    assert np.allclose(K_g_zero, K_g_zero.T)
    K_g_tension = fcn(L, A, I_rho, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.all(np.diag(K_g_tension)[1:3] > 0)
    K_g_compress = fcn(L, A, I_rho, -1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.all(np.diag(K_g_compress)[1:3] < 0)
    K_g_moment = fcn(L, A, I_rho, 0.0, 100.0, 50.0, 50.0, 50.0, 50.0)
    assert not np.allclose(K_g_moment, np.zeros((12, 12)))
    K_g_1 = fcn(L, A, I_rho, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_g_2 = fcn(L, A, I_rho, 2000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(2 * K_g_1, K_g_2)

def test_euler_buckling_cantilever_column(fcn):
    """Test geometric stiffness matrix against analytical Euler buckling load."""
    import numpy as np
    L = 3.0
    E = 200000000000.0
    I = 1e-06
    A = 0.001
    I_rho = 2e-06
    P_cr_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
    K_e = np.zeros((12, 12))
    K_e[1, 1] = K_e[7, 7] = 12 * E * I / L ** 3
    K_e[1, 5] = K_e[5, 1] = 6 * E * I / L ** 2
    K_e[1, 7] = K_e[7, 1] = -12 * E * I / L ** 3
    K_e[1, 11] = K_e[11, 1] = 6 * E * I / L ** 2
    K_e[5, 5] = K_e[11, 11] = 4 * E * I / L
    K_e[5, 7] = K_e[7, 5] = -6 * E * I / L ** 2
    K_e[5, 11] = K_e[11, 5] = 2 * E * I / L
    K_e[7, 11] = K_e[11, 7] = -6 * E * I / L ** 2
    K_g = fcn(L, A, I_rho, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    reduced_K_e = K_e[1:, 1:]
    reduced_K_g = K_g[1:, 1:]
    eigenvals = np.linalg.eigvals(reduced_K_e, -reduced_K_g)
    P_cr_numerical = np.min(np.abs(eigenvals[eigenvals != 0]))
    assert abs(P_cr_numerical - P_cr_analytical) / P_cr_analytical < 0.1