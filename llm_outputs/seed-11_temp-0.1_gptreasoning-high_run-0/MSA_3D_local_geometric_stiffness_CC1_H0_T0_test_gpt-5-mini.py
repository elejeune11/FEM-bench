def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 2.0
    A = 0.01
    I_rho = 2e-06
    K0 = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert isinstance(K0, np.ndarray)
    assert K0.shape == (12, 12)
    assert np.allclose(K0, K0.T, atol=1e-12)
    assert np.allclose(K0, np.zeros((12, 12)), atol=1e-12)
    K_t = fcn(L, A, I_rho, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_c = fcn(L, A, I_rho, -1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(K_t, -K_c, rtol=1e-09, atol=1e-12)
    rng = np.random.RandomState(0)
    v = rng.randn(12)
    q0 = float(v @ K0 @ v)
    qt = float(v @ K_t @ v)
    qc = float(v @ K_c @ v)
    assert abs(q0) <= 1e-12
    assert qt > 0.0
    assert qc < 0.0
    K100 = fcn(L, A, I_rho, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    K200 = fcn(L, A, I_rho, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(K200, 2.0 * K100, rtol=1e-09, atol=1e-12)
    K_mom = fcn(L, A, I_rho, 0.0, 12.34, 1.2, 2.3, 3.4, 4.5)
    assert not np.allclose(K_mom, K0)
    assert abs(K_mom[3, 3]) > 1e-12 or abs(K_mom[9, 9]) > 1e-12

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Tolerances account for discretization and numerical approximation.
    """
    L = 2.0
    A = 0.01
    I_rho = 2e-06
    I_y = I_rho / 2.0
    I_z = I_rho / 2.0
    J = I_rho
    E = 210000000000.0
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    K_e = np.zeros((12, 12), dtype=float)
    k_ax = E * A / L
    K_e[0, 0] = k_ax
    K_e[6, 6] = k_ax
    K_e[0, 6] = -k_ax
    K_e[6, 0] = -k_ax
    k_tor = G * J / L
    K_e[3, 3] = k_tor
    K_e[9, 9] = k_tor
    K_e[3, 9] = -k_tor
    K_e[9, 3] = -k_tor
    idx_by = [2, 4, 8, 10]
    L2 = L * L
    L3 = L2 * L
    kb_base = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float)
    kby = E * I_y / L3 * kb_base
    for (i_local, i_global) in enumerate(idx_by):
        for (j_local, j_global) in enumerate(idx_by):
            K_e[i_global, j_global] += kby[i_local, j_local]
    idx_bz = [1, 5, 7, 11]
    kbz = E * I_z / L3 * kb_base
    for (i_local, i_global) in enumerate(idx_bz):
        for (j_local, j_global) in enumerate(idx_bz):
            K_e[i_global, j_global] += kbz[i_local, j_local]
    K_g_unit = fcn(L, A, I_rho, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    free = list(range(6, 12))
    K_e_red = K_e[np.ix_(free, free)]
    K_g_unit_red = K_g_unit[np.ix_(free, free)]
    pinv_Kg = np.linalg.pinv(K_g_unit_red)
    mat = pinv_Kg @ K_e_red
    eigvals = np.linalg.eigvals(mat)
    eigvals_real = np.real(eigvals)
    positive = eigvals_real[eigvals_real > 1e-08]
    assert positive.size > 0
    P_computed = float(np.min(positive))
    P_analytic = np.pi ** 2 * E * I_y / (4.0 * L * L)
    rel_err = abs(P_computed - P_analytic) / (abs(P_analytic) + 1e-20)
    assert rel_err < 0.1