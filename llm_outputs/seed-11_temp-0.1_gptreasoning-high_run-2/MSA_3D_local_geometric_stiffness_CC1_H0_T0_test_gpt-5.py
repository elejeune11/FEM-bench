def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 3.0
    A = 0.02
    I_rho = 0.005
    K0 = fcn(L=L, A=A, I_rho=I_rho, Fx2=0.0, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    assert isinstance(K0, np.ndarray)
    assert K0.shape == (12, 12)
    assert np.allclose(K0, K0.T, atol=1e-12, rtol=1e-12)
    assert np.allclose(K0, np.zeros((12, 12)), atol=1e-12, rtol=0.0)
    F = 1234.5
    K_t = fcn(L=L, A=A, I_rho=I_rho, Fx2=+F, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    K_c = fcn(L=L, A=A, I_rho=I_rho, Fx2=-F, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    assert np.allclose(K_t, K_t.T, atol=1e-12, rtol=1e-12)
    assert np.allclose(K_c, K_c.T, atol=1e-12, rtol=1e-12)
    assert np.allclose(K_t, -K_c, rtol=1e-12, atol=1e-12)
    idx_v_thetaz = [1, 5, 7, 11]
    idx_w_thetay = [2, 4, 8, 10]
    for idxs in (idx_v_thetaz, idx_w_thetay):
        Kt_sub = K_t[np.ix_(idxs, idxs)]
        Kc_sub = K_c[np.ix_(idxs, idxs)]
        tol_t = 1e-10 * max(1.0, np.linalg.norm(Kt_sub))
        tol_c = 1e-10 * max(1.0, np.linalg.norm(Kc_sub))
        ev_t = np.linalg.eigvalsh(Kt_sub)
        ev_c = np.linalg.eigvalsh(Kc_sub)
        assert ev_t.min() >= -tol_t
        assert ev_c.max() <= tol_c
    K_mx = fcn(L=L, A=A, I_rho=I_rho, Fx2=0.0, Mx2=1000.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    K_bm = fcn(L=L, A=A, I_rho=I_rho, Fx2=0.0, Mx2=0.0, My1=300.0, Mz1=-200.0, My2=100.0, Mz2=50.0)
    assert np.allclose(K_mx, K_mx.T, atol=1e-12, rtol=1e-12)
    assert np.allclose(K_bm, K_bm.T, atol=1e-12, rtol=1e-12)
    assert not np.allclose(K_mx, K0, atol=1e-12, rtol=1e-12)
    assert not np.allclose(K_bm, K0, atol=1e-12, rtol=1e-12)
    assert not np.allclose(K_mx, K_bm, atol=1e-12, rtol=1e-12)
    F1 = 200.0
    alpha = 3.7
    K_F1 = fcn(L=L, A=A, I_rho=I_rho, Fx2=F1, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    K_aF1 = fcn(L=L, A=A, I_rho=I_rho, Fx2=alpha * F1, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    assert np.allclose(K_aF1, alpha * K_F1, rtol=1e-12, atol=1e-12)

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load; tolerances account for discretization error.
    """
    L_total = 2.0
    E = 210000000000.0
    Iy = 8e-06
    EI = E * Iy
    n_elem = 20
    n_nodes = n_elem + 1
    dof_per_node = 2
    ndof = dof_per_node * n_nodes
    Le = L_total / n_elem

    def ke_bending(EI_val, L_el):
        L = L_el
        k = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L]])
        return EI_val / L ** 3 * k
    A = 1.0
    I_rho = 1.0
    Kge_full = fcn(L=Le, A=A, I_rho=I_rho, Fx2=-1.0, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    idx_v_thetaz = [1, 5, 7, 11]
    kge = Kge_full[np.ix_(idx_v_thetaz, idx_v_thetaz)]
    K_e = np.zeros((ndof, ndof))
    K_g_unit = np.zeros((ndof, ndof))
    ke = ke_bending(EI, Le)
    for e in range(n_elem):
        i = e
        j = e + 1
        dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        K_e[np.ix_(dofs, dofs)] += ke
        K_g_unit[np.ix_(dofs, dofs)] += kge
    free_dofs = list(range(2, ndof))
    K_e_red = K_e[np.ix_(free_dofs, free_dofs)]
    K_g_red = K_g_unit[np.ix_(free_dofs, free_dofs)]
    M = np.linalg.solve(-K_g_red, K_e_red)
    evals = np.linalg.eigvals(M)
    evals_real = np.real(evals[np.isreal(evals)])
    pos_evals = evals_real[evals_real > 0.0]
    assert pos_evals.size > 0
    Pcr_numeric = np.min(pos_evals)
    Pcr_exact = np.pi ** 2 * EI / (4.0 * L_total ** 2)
    rel_err = abs(Pcr_numeric - Pcr_exact) / Pcr_exact
    assert rel_err < 0.05, f'Relative error too large: {rel_err}'