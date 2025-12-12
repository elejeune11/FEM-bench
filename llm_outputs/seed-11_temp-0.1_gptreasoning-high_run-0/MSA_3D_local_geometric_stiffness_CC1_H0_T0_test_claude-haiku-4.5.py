def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 10.0
    A = 1.0
    I_rho = 0.1
    Kg = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert Kg.shape == (12, 12), 'Matrix shape should be 12x12'
    assert np.allclose(Kg, Kg.T, atol=1e-10), 'Geometric stiffness matrix should be symmetric'
    Kg_zero = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(Kg_zero, np.zeros((12, 12)), atol=1e-12), 'Matrix should be zero when all loads are zero'
    Kg_tension = fcn(L, A, I_rho, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    Kg_no_load = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    lateral_indices = [1, 2, 4, 5, 7, 8, 10, 11]
    for i in lateral_indices:
        for j in lateral_indices:
            assert Kg_tension[i, j] > Kg_no_load[i, j], f'Tension should increase stiffness at ({i},{j})'
    Kg_compression = fcn(L, A, I_rho, -100.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for i in lateral_indices:
        for j in lateral_indices:
            assert Kg_compression[i, j] < Kg_no_load[i, j], f'Compression should decrease stiffness at ({i},{j})'
    Kg_no_mx = fcn(L, A, I_rho, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    Kg_with_mx = fcn(L, A, I_rho, 50.0, 10.0, 0.0, 0.0, 0.0, 0.0)
    assert not np.allclose(Kg_no_mx, Kg_with_mx, atol=1e-10), 'Matrix should change when torsional moment is applied'
    Kg_no_my = fcn(L, A, I_rho, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    Kg_with_my = fcn(L, A, I_rho, 50.0, 0.0, 5.0, 0.0, 0.0, 0.0)
    assert not np.allclose(Kg_no_my, Kg_with_my, atol=1e-10), 'Matrix should change when bending moment My1 is applied'
    Kg_no_mz = fcn(L, A, I_rho, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    Kg_with_mz = fcn(L, A, I_rho, 50.0, 0.0, 0.0, 5.0, 0.0, 0.0)
    assert not np.allclose(Kg_no_mz, Kg_with_mz, atol=1e-10), 'Matrix should change when bending moment Mz1 is applied'
    Kg_fx_50 = fcn(L, A, I_rho, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    Kg_fx_100 = fcn(L, A, I_rho, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    Kg_fx_150 = fcn(L, A, I_rho, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    diff_1 = Kg_fx_100 - Kg_fx_50
    diff_2 = Kg_fx_150 - Kg_fx_100
    assert np.allclose(diff_1, diff_2, atol=1e-10), 'Matrix should scale linearly with Fx2'

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to the correct 
    buckling load for a cantilever column. Compare numerical result with the analytical 
    Euler buckling load. Design the test so that comparison tolerances account for 
    discretization error.
    """
    L = 10.0
    A = 1.0
    I = 0.1
    E = 1.0
    P_cr_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
    K_e_22 = np.array([[12 * E * I / L ** 3, 6 * E * I / L ** 2], [6 * E * I / L ** 2, 4 * E * I / L]])
    Kg_full = fcn(L, A, I, -P_cr_analytical, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_g_22 = Kg_full[np.ix_([1, 5], [1, 5])]
    K_combined = K_e_22 + K_g_22
    det_combined = np.linalg.det(K_combined)
    assert abs(det_combined) < 1e-06, f'Determinant at analytical buckling load should be near zero, got {det_combined}'
    P_below = 0.8 * P_cr_analytical
    Kg_below = fcn(L, A, I, -P_below, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_g_below_22 = Kg_below[np.ix_([1, 5], [1, 5])]
    K_combined_below = K_e_22 + K_g_below_22
    det_below = np.linalg.det(K_combined_below)
    assert det_below > 0, f'System should be stable below critical load, det = {det_below}'
    P_above = 1.2 * P_cr_analytical
    Kg_above = fcn(L, A, I, -P_above, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_g_above_22 = Kg_above[np.ix_([1, 5], [1, 5])]
    K_combined_above = K_e_22 + K_g_above_22
    det_above = np.linalg.det(K_combined_above)
    assert det_above < 0, f'System should be unstable above critical load, det = {det_above}'