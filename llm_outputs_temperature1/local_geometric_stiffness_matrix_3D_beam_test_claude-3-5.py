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
    K_g_compress = fcn(L, A, I_rho, -1000, 0, 0, 0, 0, 0)
    assert not np.allclose(K_g_tension, K_g_compress)
    K_g_base = fcn(L, A, I_rho, 1000, 0, 0, 0, 0, 0)
    K_g_moment = fcn(L, A, I_rho, 1000, 100, 200, 300, 400, 500)
    assert not np.allclose(K_g_base, K_g_moment)
    K_g_1 = fcn(L, A, I_rho, 1000, 0, 0, 0, 0, 0)
    K_g_2 = fcn(L, A, I_rho, 2000, 0, 0, 0, 0, 0)
    assert np.allclose(2 * K_g_1, K_g_2)

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to 
    the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Design the test so that comparison tolerances account for discretization error.
    """
    import numpy as np
    from scipy.linalg import eigh
    L = 3.0
    E = 200000000000.0
    I = 1e-06
    A = 0.01
    I_rho = 2e-06
    P_cr_analytical = -np.pi ** 2 * E * I / (4 * L ** 2)
    K_e = np.zeros((12, 12))
    K_e[1, 1] = K_e[7, 7] = 12 * E * I / L ** 3
    K_e[1, 5] = K_e[5, 1] = 6 * E * I / L ** 2
    K_e[1, 7] = K_e[7, 1] = -12 * E * I / L ** 3
    K_e[1, 11] = K_e[11, 1] = 6 * E * I / L ** 2
    K_e[5, 5] = K_e[11, 11] = 4 * E * I / L
    K_e[5, 7] = K_e[7, 5] = -6 * E * I / L ** 2
    K_e[5, 11] = K_e[11, 5] = 2 * E * I / L
    K_e[7, 11] = K_e[11, 7] = -6 * E * I / L ** 2
    P_unit = -1.0
    K_g = fcn(L, A, I_rho, P_unit, 0, 0, 0, 0, 0)
    (eigenvals, _) = eigh(K_e, K_g)
    P_cr_numerical = eigenvals[eigenvals > 0].min()
    assert abs(P_cr_numerical - abs(P_cr_analytical)) / abs(P_cr_analytical) < 0.1