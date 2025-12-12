def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    L = 10.0
    A = 0.01
    I_rho = 1e-06
    Fx2 = 0.0
    Mx2 = 0.0
    My1 = 0.0
    Mz1 = 0.0
    My2 = 0.0
    Mz2 = 0.0
    K0 = fcn(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
    assert K0.shape == (12, 12)
    assert np.allclose(K0, K0.T)
    assert np.allclose(K0, np.zeros((12, 12)))
    Fx2_pos = 1000.0
    K_pos = fcn(L, A, I_rho, Fx2_pos, Mx2, My1, Mz1, My2, Mz2)
    assert not np.allclose(K_pos, K0)
    assert np.all(K_pos >= K0 - 1e-12)
    Fx2_neg = -1000.0
    K_neg = fcn(L, A, I_rho, Fx2_neg, Mx2, My1, Mz1, My2, Mz2)
    assert not np.allclose(K_neg, K0)
    assert np.any(K_neg < K0 - 1e-12)
    Mx2_val = 50.0
    My1_val = 30.0
    Mz1_val = 20.0
    My2_val = 40.0
    Mz2_val = 60.0
    K_moments = fcn(L, A, I_rho, Fx2, Mx2_val, My1_val, Mz1_val, My2_val, Mz2_val)
    assert not np.allclose(K_moments, K0)
    K_double = fcn(L, A, I_rho, 2 * Fx2_pos, Mx2, My1, Mz1, My2, Mz2)
    assert np.allclose(K_double, 2 * K_pos, atol=1e-10)

def test_euler_buckling_cantilever_column(fcn):
    L = 1.0
    A = 1.0
    I_rho = 1.0
    E = 200000000000.0
    I = 1.0
    Fx2 = -1.0
    Mx2 = 0.0
    My1 = 0.0
    Mz1 = 0.0
    My2 = 0.0
    Mz2 = 0.0
    K_geo = fcn(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
    K_elastic = np.eye(12) * 1e-10
    P_euler = np.pi ** 2 * E * I / (4 * L ** 2)
    Fx2_double = 2 * Fx2
    K_geo_double = fcn(L, A, I_rho, Fx2_double, Mx2, My1, Mz1, My2, Mz2)
    assert np.allclose(K_geo_double, 2 * K_geo, atol=1e-10)
    eigenvals = np.linalg.eigvalsh(K_geo)
    assert np.any(eigenvals < 0)
    min_eig = np.min(eigenvals)
    eigenvals_double = np.linalg.eigvalsh(K_geo_double)
    min_eig_double = np.min(eigenvals_double)
    assert np.allclose(min_eig_double, 2 * min_eig, rtol=1e-10)