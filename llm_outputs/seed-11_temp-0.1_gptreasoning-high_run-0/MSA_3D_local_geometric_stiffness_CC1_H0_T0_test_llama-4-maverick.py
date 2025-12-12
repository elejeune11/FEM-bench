def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    (L, A, I_rho) = (1.0, 1.0, 1.0)
    (Fx2, Mx2, My1, Mz1, My2, Mz2) = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    K_g = fcn(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
    assert K_g.shape == (12, 12)
    assert np.allclose(K_g, K_g.T)
    assert np.allclose(fcn(L, A, I_rho, 0, 0, 0, 0, 0, 0), np.zeros((12, 12)))
    K_g_tension = fcn(L, A, I_rho, 1.0, Mx2, My1, Mz1, My2, Mz2)
    K_g_compression = fcn(L, A, I_rho, -1.0, Mx2, My1, Mz1, My2, Mz2)
    assert np.all(K_g_tension.diagonal() >= 0)
    assert np.any(K_g_compression.diagonal() < 0)
    K_g_var1 = fcn(L, A, I_rho, Fx2, 2.0, My1, Mz1, My2, Mz2)
    K_g_var2 = fcn(L, A, I_rho, Fx2, Mx2, 2.0 * My1, Mz1, My2, Mz2)
    assert not np.allclose(K_g, K_g_var1)
    assert not np.allclose(K_g, K_g_var2)
    K_g_scaled = fcn(L, A, I_rho, 2.0 * Fx2, Mx2, My1, Mz1, My2, Mz2)
    assert np.allclose(K_g_scaled, 2.0 * K_g)

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Design the test so that comparison tolerances account for discretization error.
    """
    (L, A, I_rho) = (10.0, 1.0, 2.0)
    E = 200000000000.0
    I = 1.0
    n_elements = 10
    L_element = L / n_elements
    K_e_local = np.zeros((12, 12))
    K_g_local = np.zeros((12, 12))
    K_e_global = np.zeros((6 * n_elements + 6, 6 * n_elements + 6))
    K_g_global = np.zeros((6 * n_elements + 6, 6 * n_elements + 6))
    for i in range(n_elements):
        K_e_local[:6, :6] = np.array([[E * A / L_element, 0, 0, 0, 0, 0], [0, 12 * E * I / L_element ** 3, 0, 0, 0, 6 * E * I / L_element ** 2], [0, 0, 12 * E * I / L_element ** 3, 0, -6 * E * I / L_element ** 2, 0], [0, 0, 0, E * I_rho / L_element, 0, 0], [0, 0, -6 * E * I / L_element ** 2, 0, 4 * E * I / L_element, 0], [0, 6 * E * I / L_element ** 2, 0, 0, 0, 4 * E * I / L_element]])
        K_e_local[6:, 6:] = K_e_local[:6, :6]
        K_e_local[:6, 6:] = np.array([[-E * A / L_element, 0, 0, 0, 0, 0], [0, -12 * E * I / L_element ** 3, 0, 0, 0, 6 * E * I / L_element ** 2], [0, 0, -12 * E * I / L_element ** 3, 0, -6 * E * I / L_element ** 2, 0], [0, 0, 0, -E * I_rho / L_element, 0, 0], [0, 0, 6 * E * I / L_element ** 2, 0, 2 * E * I / L_element, 0], [0, -6 * E * I / L_element ** 2, 0, 0, 0, 2 * E * I / L_element]])
        K_e_local[6:, :6] = K_e_local[:6, 6:].T
        K_g_local = fcn(L_element, A, I_rho, -1.0, 0, 0, 0, 0, 0)
        idx = [6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * (i + 1), 6 * (i + 1) + 1, 6 * (i + 1) + 2, 6 * (i + 1) + 3, 6 * (i + 1) + 4, 6 * (i + 1) + 5]
        K_e_global[np.ix_(idx, idx)] += K_e_local
        K_g_global[np.ix_(idx, idx)] += K_g_local
    fixed_dofs = [0, 1, 2, 3, 4, 5]
    free_dofs = np.setdiff1d(np.arange(6 * n_elements + 6), fixed_dofs)
    K_e_ff = K_e_global[np.ix_(free_dofs, free_dofs)]
    K_g_ff = K_g_global[np.ix_(free_dofs, free_dofs)]
    (w, v) = np.linalg.eig(np.linalg.solve(K_e_ff, -K_g_ff))
    P_cr_num = np.min(np.abs(w))
    P_cr_exact = (np.pi / 2) ** 2 * E * I / L ** 2
    assert np.isclose(P_cr_num, P_cr_exact, rtol=0.01)