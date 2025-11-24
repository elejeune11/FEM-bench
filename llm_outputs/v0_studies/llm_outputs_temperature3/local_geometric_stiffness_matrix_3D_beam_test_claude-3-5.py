def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    import numpy as np
    L = 2.0
    A = 0.01
    I_rho = 0.0001
    K_g = fcn(L, A, I_rho, 0, 0, 0, 0, 0, 0)
    assert K_g.shape == (12, 12)
    assert np.allclose(K_g, np.zeros((12, 12)))
    K_g = fcn(L, A, I_rho, 1000, 100, 200, 300, 400, 500)
    assert np.allclose(K_g, K_g.T)
    K_g_tension = fcn(L, A, I_rho, 1000, 0, 0, 0, 0, 0)
    K_g_compression = fcn(L, A, I_rho, -1000, 0, 0, 0, 0, 0)
    assert not np.allclose(K_g_tension, K_g_compression)
    K_g_base = fcn(L, A, I_rho, 1000, 0, 0, 0, 0, 0)
    K_g_moment = fcn(L, A, I_rho, 1000, 100, 200, 300, 400, 500)
    assert not np.allclose(K_g_base, K_g_moment)
    K_g_1 = fcn(L, A, I_rho, 1000, 0, 0, 0, 0, 0)
    K_g_2 = fcn(L, A, I_rho, 2000, 0, 0, 0, 0, 0)
    assert np.allclose(2 * K_g_1, K_g_2)

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load
    for a cantilever column. Compare numerical result with the analytical Euler buckling load.
    """
    import numpy as np
    E = 200000000000.0
    L = 3.0
    b = 0.1
    h = 0.1
    A = b * h
    I = b * h ** 3 / 12
    I_rho = 2 * I
    P_cr_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
    k11 = 12 * E * I / L ** 3
    k12 = 6 * E * I / L ** 2
    k22 = 4 * E * I / L
    K_e = np.zeros((12, 12))
    K_e[1, 1] = k11
    K_e[1, 5] = -k12
    K_e[5, 1] = -k12
    K_e[5, 5] = k22

    def det_K(P):
        K_g = fcn(L, A, I_rho, -P, 0, 0, 0, 0, 0)
        return np.linalg.det(K_e + K_g)
    P_low = 0
    P_high = 2 * P_cr_analytical
    tol = 0.001
    while P_high - P_low > tol * P_cr_analytical:
        P_mid = (P_low + P_high) / 2
        if det_K(P_mid) > 0:
            P_low = P_mid
        else:
            P_high = P_mid
    P_cr_numerical = (P_low + P_high) / 2
    assert abs(P_cr_numerical - P_cr_analytical) / P_cr_analytical < 0.05