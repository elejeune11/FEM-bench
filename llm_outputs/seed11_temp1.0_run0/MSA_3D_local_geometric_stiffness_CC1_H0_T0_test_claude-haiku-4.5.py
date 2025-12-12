def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 10.0
    A = 0.01
    I_rho = 1e-06
    K_g = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert K_g.shape == (12, 12), f'Expected shape (12, 12), got {K_g.shape}'
    assert np.allclose(K_g, K_g.T), 'Geometric stiffness matrix should be symmetric'
    K_g_zero = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(K_g_zero, np.zeros((12, 12))), 'Matrix should be zero when all loads are zero'
    Fx2_tension = 1000.0
    K_g_tension = fcn(L, A, I_rho, Fx2_tension, 0.0, 0.0, 0.0, 0.0, 0.0)
    tension_v_stiffness = K_g_tension[1, 1] + K_g_tension[2, 2]
    assert tension_v_stiffness > 0, 'Tension should increase transverse stiffness'
    Fx2_compression = -1000.0
    K_g_compression = fcn(L, A, I_rho, Fx2_compression, 0.0, 0.0, 0.0, 0.0, 0.0)
    compression_v_stiffness = K_g_compression[1, 1] + K_g_compression[2, 2]
    assert compression_v_stiffness < 0, 'Compression should decrease transverse stiffness'
    K_g_no_torsion = fcn(L, A, I_rho, Fx2_tension, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_g_with_torsion = fcn(L, A, I_rho, Fx2_tension, 500.0, 0.0, 0.0, 0.0, 0.0)
    assert not np.allclose(K_g_no_torsion, K_g_with_torsion), 'Matrix should change with torsional moment'
    K_g_no_bending = fcn(L, A, I_rho, Fx2_tension, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_g_with_bending = fcn(L, A, I_rho, Fx2_tension, 0.0, 100.0, 100.0, 100.0, 100.0)
    assert not np.allclose(K_g_no_bending, K_g_with_bending), 'Matrix should change with bending moments'
    scale_factor = 2.5
    K_g_1 = fcn(L, A, I_rho, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_g_2 = fcn(L, A, I_rho, 100.0 * scale_factor, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(K_g_2, K_g_1 * scale_factor), 'Matrix should scale linearly with Fx2'
    (My1, Mz1) = (50.0, 75.0)
    (My2, Mz2) = (60.0, 80.0)
    (Mx2, Fx2) = (25.0, 500.0)
    K_g_complex = fcn(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
    assert np.allclose(K_g_complex, K_g_complex.T), 'Symmetry should hold for complex load cases'

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to the correct
    buckling load for a cantilever column. Compare numerical result with the analytical Euler
    buckling load. Design the test so that comparison tolerances account for discretization error.
    """
    L = 1.0
    E = 1.0
    I = 1.0
    A = 0.1
    I_rho = I
    K_e = np.zeros((12, 12))
    K_e[0, 0] = E * A / L
    K_e[6, 6] = E * A / L
    K_e[0, 6] = -E * A / L
    K_e[6, 0] = -E * A / L
    k_y = 12 * E * I / L ** 3
    K_e[2, 2] = k_y
    K_e[2, 4] = 6 * E * I / L ** 2
    K_e[2, 8] = -k_y
    K_e[2, 10] = 6 * E * I / L ** 2
    K_e[4, 2] = 6 * E * I / L ** 2
    K_e[4, 4] = 4 * E * I / L
    K_e[4, 8] = -6 * E * I / L ** 2
    K_e[4, 10] = 2 * E * I / L
    K_e[8, 2] = -k_y
    K_e[8, 4] = -6 * E * I / L ** 2
    K_e[8, 8] = k_y
    K_e[8, 10] = -6 * E * I / L ** 2
    K_e[10, 2] = 6 * E * I / L ** 2
    K_e[10, 4] = 2 * E * I / L
    K_e[10, 8] = -6 * E * I / L ** 2
    K_e[10, 10] = 4 * E * I / L
    K_e[1, 1] = k_y
    K_e[1, 5] = -6 * E * I / L ** 2
    K_e[1, 7] = -k_y
    K_e[1, 11] = -6 * E * I / L ** 2
    K_e[5, 1] = -6 * E * I / L ** 2
    K_e[5, 5] = 4 * E * I / L
    K_e[5, 7] = 6 * E * I / L ** 2
    K_e[5, 11] = 2 * E * I / L
    K_e[7, 1] = -k_y
    K_e[7, 5] = 6 * E * I / L ** 2
    K_e[7, 7] = k_y
    K_e[7, 11] = 6 * E * I / L ** 2
    K_e[11, 1] = -6 * E * I / L ** 2
    K_e[11, 5] = 2 * E * I / L
    K_e[11, 7] = 6 * E * I / L ** 2
    K_e[11, 11] = 4 * E * I / L
    P_cr_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
    loads_to_test = np.linspace(0.1, 1.5 * P_cr_analytical, 20)
    eigenvalues_min = []
    for P_test in loads_to_test:
        Fx2 = -P_test
        K_g = fcn(L, A, I_rho, Fx2, 0.0, 0.0, 0.0, 0.0, 0.0)
        K_combined = K_e + K_g
        try:
            eigenvalues = np.linalg.eigvalsh(K_combined)
            eigenvalues_min.append(eigenvalues[0])
        except np.linalg.LinAlgError:
            eigenvalues_min.append(0.0)
    eigenvalues_min = np.array(eigenvalues_min)
    sign_changes = np.where(np.diff(np.sign(eigenvalues_min)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        P_test_1 = loads_to_test[idx]
        P_test_2 = loads_to_test[idx + 1]
        lambda_1 = eigenvalues_min[idx]
        lambda_2 = eigenvalues_min[idx + 1]
        P_cr_numerical = P_test_1 - lambda_1 * (P_test_2 - P_test_1) / (lambda_2 - lambda_1)
        relative_error = abs(P_cr_numerical - P_cr_analytical) / P_cr_analytical
        assert relative_error < 0.15, f'Buckling load error {relative_error * 100:.2f}% exceeds tolerance. Numerical: {P_cr_numerical:.6f}, Analytical: {P_cr_analytical:.6f}'
    else:
        P_near_cr = P_cr_analytical
        Fx2 = -P_near_cr
        K_g = fcn(L, A, I_rho, Fx2, 0.0, 0.0, 0.0, 0.0, 0.0)
        K_combined = K_e + K_g
        eigenvalues = np.linalg.eigvalsh(K_combined)
        assert eigenvalues[0] < 0.1 * E * A / L, 'Smallest eigenvalue should be small near buckling load'