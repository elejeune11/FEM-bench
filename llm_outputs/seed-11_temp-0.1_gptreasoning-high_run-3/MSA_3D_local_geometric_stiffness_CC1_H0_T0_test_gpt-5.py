def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 2.5
    A = 0.03
    I_rho = 0.01
    K_zero = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert isinstance(K_zero, np.ndarray)
    assert K_zero.shape == (12, 12)
    assert np.allclose(K_zero, K_zero.T, atol=1e-12, rtol=0.0)
    assert np.allclose(K_zero, np.zeros((12, 12)), atol=1e-12, rtol=0.0)
    P = 5.0
    K_tension = fcn(L, A, I_rho, P, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_compression = fcn(L, A, I_rho, -P, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(K_tension, K_tension.T, atol=1e-12, rtol=0.0)
    assert np.allclose(K_compression, K_compression.T, atol=1e-12, rtol=0.0)
    assert not np.allclose(K_tension, np.zeros((12, 12)), atol=1e-12, rtol=0.0)
    q = np.arange(1, 13, dtype=float)
    stiff_tension = float(q @ K_tension @ q)
    stiff_compression = float(q @ K_compression @ q)
    assert stiff_tension > 0.0
    assert stiff_compression < 0.0
    assert np.isclose(stiff_compression, -stiff_tension, rtol=1e-12, atol=1e-12)
    assert np.allclose(K_compression, -K_tension, rtol=1e-12, atol=1e-12)
    K_Mx = fcn(L, A, I_rho, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0)
    K_My1 = fcn(L, A, I_rho, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0)
    K_Mz1 = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0)
    K_My2 = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, -2.5, 0.0)
    K_Mz2 = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, -1.2)
    assert not np.allclose(K_Mx, np.zeros((12, 12)), atol=1e-12, rtol=0.0)
    assert not np.allclose(K_My1, np.zeros((12, 12)), atol=1e-12, rtol=0.0)
    assert not np.allclose(K_Mz1, np.zeros((12, 12)), atol=1e-12, rtol=0.0)
    assert not np.allclose(K_My2, np.zeros((12, 12)), atol=1e-12, rtol=0.0)
    assert not np.allclose(K_Mz2, np.zeros((12, 12)), atol=1e-12, rtol=0.0)
    assert not np.allclose(K_My1, K_My2, rtol=1e-10, atol=1e-12)
    assert not np.allclose(K_Mz1, K_Mz2, rtol=1e-10, atol=1e-12)
    assert not np.allclose(K_Mx, K_My1, rtol=1e-10, atol=1e-12)
    assert not np.allclose(K_Mx, K_Mz1, rtol=1e-10, atol=1e-12)
    Fx = 1.2345
    K1 = fcn(L, A, I_rho, Fx, 0.0, 0.0, 0.0, 0.0, 0.0)
    K3 = fcn(L, A, I_rho, 3.0 * Fx, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(K3, 3.0 * K1, rtol=1e-12, atol=1e-12)

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Design the test so that comparison tolerances account for discretization error.
    """
    L = 1.0
    A = 1.0
    I_rho = 1.0
    E = 1.0
    Iz = 1.0
    n_elems = 30
    n_nodes = n_elems + 1
    dof_per_node_sub = 2
    n_dof_sub = dof_per_node_sub * n_nodes
    K_e_sub = np.zeros((n_dof_sub, n_dof_sub), dtype=float)
    K_g_sub = np.zeros((n_dof_sub, n_dof_sub), dtype=float)

    def bending_elastic_block(EI, Le):
        return EI / Le ** 3 * np.array([[12.0, 6.0 * Le, -12.0, 6.0 * Le], [6.0 * Le, 4.0 * Le ** 2, -6.0 * Le, 2.0 * Le ** 2], [-12.0, -6.0 * Le, 12.0, -6.0 * Le], [6.0 * Le, 2.0 * Le ** 2, -6.0 * Le, 4.0 * Le ** 2]])
    for e in range(n_elems):
        Le = L / n_elems
        ke_loc = bending_elastic_block(E * Iz, Le)
        Kg_full_unit = fcn(Le, A, I_rho, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        idx_full = [1, 5, 7, 11]
        kg_loc = Kg_full_unit[np.ix_(idx_full, idx_full)]
        idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        for ii in range(4):
            for jj in range(4):
                K_e_sub[idx[ii], idx[jj]] += ke_loc[ii, jj]
                K_g_sub[idx[ii], idx[jj]] += kg_loc[ii, jj]
    fixed = [0, 1]
    free = [i for i in range(n_dof_sub) if i not in fixed]
    Ke_r = K_e_sub[np.ix_(free, free)]
    Kg_r = K_g_sub[np.ix_(free, free)]
    Lchol = np.linalg.cholesky(Kg_r)
    Y = np.linalg.solve(Lchol, Ke_r)
    C = np.linalg.solve(Lchol.T, Y.T).T
    eigvals = np.linalg.eigvalsh((C + C.T) * 0.5)
    eigvals_real = eigvals[np.isfinite(eigvals)]
    eigvals_positive = eigvals_real[eigvals_real > 1e-10]
    Pcr_num = float(np.min(eigvals_positive))
    Pcr_analytical = np.pi ** 2 * E * Iz / (4.0 * L ** 2)
    assert np.isclose(Pcr_num, Pcr_analytical, rtol=0.03, atol=1e-08)