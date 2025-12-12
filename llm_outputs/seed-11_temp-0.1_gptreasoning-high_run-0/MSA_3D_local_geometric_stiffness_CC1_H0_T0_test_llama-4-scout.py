def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for MSA_3D_local_geometric_stiffness_CC1_H0_T0:
    """
    (L, A, I_rho) = (1.0, 0.1, 0.01)
    (Fx2, Mx2, My1, Mz1, My2, Mz2) = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_g = fcn(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
    assert K_g.shape == (12, 12)
    assert np.allclose(K_g, K_g.T)
    (Fx2, Mx2, My1, Mz1, My2, Mz2) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_g = fcn(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
    assert np.allclose(K_g, np.zeros((12, 12)))
    Fx2_values = [1.0, -1.0]
    for Fx2 in Fx2_values:
        K_g = fcn(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        assert np.linalg.det(K_g) * np.linalg.det(np.eye(12)) < 0 if Fx2 < 0 else True
    Mx2_values = [1.0, -1.0]
    for Mx2 in Mx2_values:
        K_g1 = fcn(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        K_g2 = fcn(L, A, I_rho, Fx2, -Mx2, My1, Mz1, My2, Mz2)
        assert not np.allclose(K_g1, K_g2)
    Fx2_values = [1.0, 2.0, 3.0]
    K_g_matrices = []
    for Fx2 in Fx2_values:
        K_g = fcn(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        K_g_matrices.append(K_g)
    for i in range(1, len(Fx2_values)):
        assert np.allclose(K_g_matrices[i] / Fx2_values[i], K_g_matrices[0] / Fx2_values[0])

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to the 
    correct buckling load for a cantilever column. Compare numerical result with the 
    analytical Euler buckling load. Design the test so that comparison tolerances 
    account for discretization error.
    """
    (L, A, I_rho) = (1.0, 0.1, 0.01)
    (Fx2, Mx2, My1, Mz1, My2, Mz2) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    P_cr_analytical = np.pi ** 2 * 0.1 * 0.01 / (4 * L ** 2)
    K_e = np.eye(12)
    P_values = np.linspace(0, 10 * P_cr_analytical, 100)
    for P in P_values:
        Fx2 = -P
        K_g = fcn(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        K = K_e + K_g
        eigenvalues = np.linalg.eigvals(K)
        if np.any(eigenvalues < 0):
            P_cr_numerical = -Fx2
            break
    assert np.isclose(P_cr_numerical, P_cr_analytical, rtol=0.1)