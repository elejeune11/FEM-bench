def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam verifying:
    """
    (L, A, I_rho) = (2.0, 0.01, 0.0001)
    K_g_zero = fcn(L, A, I_rho, 0, 0, 0, 0, 0, 0)
    assert K_g_zero.shape == (12, 12)
    assert_array_almost_equal(K_g_zero, np.zeros((12, 12)))
    K_g = fcn(L, A, I_rho, 1000, 100, 50, -50, -100, 100)
    assert_array_almost_equal(K_g, K_g.T)
    K_g_tension = fcn(L, A, I_rho, 1000, 0, 0, 0, 0, 0)
    K_g_compression = fcn(L, A, I_rho, -1000, 0, 0, 0, 0, 0)
    assert_array_less(K_g_compression, K_g_tension)
    K_g_base = fcn(L, A, I_rho, 1000, 0, 0, 0, 0, 0)
    K_g_moment = fcn(L, A, I_rho, 1000, 100, 50, 50, 50, 50)
    assert not np.allclose(K_g_base, K_g_moment)
    K_g_1 = fcn(L, A, I_rho, 1000, 0, 0, 0, 0, 0)
    K_g_2 = fcn(L, A, I_rho, 2000, 0, 0, 0, 0, 0)
    assert_array_almost_equal(2 * K_g_1, K_g_2)

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to 
    the correct buckling load for a cantilever column. Compare numerical result 
    with the analytical Euler buckling load.
    """
    E = 200000000000.0
    L = 3.0
    b = 0.1
    A = b * b
    I = b ** 4 / 12
    I_rho = 2 * I
    P_cr_theoretical = np.pi ** 2 * E * I / (4 * L ** 2)

    def K_e_local():
        k = np.zeros((12, 12))
        k[0, 0] = k[6, 6] = E * A / L
        k[0, 6] = k[6, 0] = -E * A / L
        EI = E * I
        k[1, 1] = k[7, 7] = 12 * EI / L ** 3
        k[1, 5] = k[5, 1] = 6 * EI / L ** 2
        k[1, 7] = k[7, 1] = -12 * EI / L ** 3
        k[1, 11] = k[11, 1] = 6 * EI / L ** 2
        k[5, 5] = k[11, 11] = 4 * EI / L
        k[5, 7] = k[7, 5] = -6 * EI / L ** 2
        k[5, 11] = k[11, 5] = 2 * EI / L
        k[7, 11] = k[11, 7] = -6 * EI / L ** 2
        k[2, 2] = k[1, 1]
        k[2, 4] = -k[1, 5]
        k[2, 8] = k[1, 7]
        k[2, 10] = -k[1, 11]
        k[4, 4] = k[5, 5]
        k[4, 8] = -k[5, 7]
        k[4, 10] = k[5, 11]
        k[8, 8] = k[7, 7]
        k[8, 10] = -k[7, 11]
        k[10, 10] = k[11, 11]
        return k

    def get_critical_load(P_test):
        K_g = fcn(L, A, I_rho, P_test, 0, 0, 0, 0, 0)
        K = K_e_local() + K_g
        K_reduced = K[6:, 6:]
        eigvals = eigh(K_reduced, eigvals_only=True)
        return np.min(eigvals[eigvals > 1e-10])
    (P_low, P_high) = (0, 2 * P_cr_theoretical)
    for _ in range(20):
        P_mid = (P_low + P_high) / 2
        if get_critical_load(P_mid) > 0:
            P_low = P_mid
        else:
            P_high = P_mid
    P_cr_numerical = P_low
    assert abs(P_cr_numerical - P_cr_theoretical) / P_cr_theoretical < 0.05