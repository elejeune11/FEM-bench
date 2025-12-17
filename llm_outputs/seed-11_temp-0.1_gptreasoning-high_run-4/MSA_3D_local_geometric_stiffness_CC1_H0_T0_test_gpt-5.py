def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 2.5
    A = 0.01
    I_rho = 1.5e-05
    K0 = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert isinstance(K0, np.ndarray)
    assert K0.shape == (12, 12)
    assert np.allclose(K0, K0.T, atol=1e-12, rtol=0.0)
    assert np.allclose(K0, np.zeros((12, 12)), atol=1e-12, rtol=0.0)
    K_generic = fcn(L, A, I_rho, 13.2, 4.1, -2.5, 1.7, 0.9, -0.4)
    assert K_generic.shape == (12, 12)
    assert np.allclose(K_generic, K_generic.T, atol=1e-10, rtol=0.0)
    P = 10.0
    K_tension = fcn(L, A, I_rho, P, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_compress = fcn(L, A, I_rho, -P, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(K_tension, K_tension.T, atol=1e-12)
    assert np.allclose(K_compress, K_compress.T, atol=1e-12)
    bend_idx = [1, 2, 4, 5, 7, 8, 10, 11]
    Kt_b = K_tension[np.ix_(bend_idx, bend_idx)]
    Kc_b = K_compress[np.ix_(bend_idx, bend_idx)]
    eig_t = np.linalg.eigvalsh((Kt_b + Kt_b.T) * 0.5)
    eig_c = np.linalg.eigvalsh((Kc_b + Kc_b.T) * 0.5)
    tol_t = 1e-12 * (np.max(np.abs(eig_t)) + 1.0)
    tol_c = 1e-12 * (np.max(np.abs(eig_c)) + 1.0)
    assert eig_t.min() >= -tol_t
    assert eig_c.max() <= tol_c
    K_mx = fcn(L, A, I_rho, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0)
    K_my1 = fcn(L, A, I_rho, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0)
    K_mz1 = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0)
    K_my2 = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0)
    K_mz2 = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0)
    assert np.linalg.norm(K_mx - K0) > 0.0
    assert np.linalg.norm(K_my1 - K0) > 0.0
    assert np.linalg.norm(K_mz1 - K0) > 0.0
    assert np.linalg.norm(K_my2 - K0) > 0.0
    assert np.linalg.norm(K_mz2 - K0) > 0.0
    alpha = 3.7
    K_P = fcn(L, A, I_rho, P, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_alphaP = fcn(L, A, I_rho, alpha * P, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(K_alphaP, alpha * K_P, rtol=1e-12, atol=1e-12)

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Design the test so that comparison tolerances account for discretization error.
    """
    E = 210000000000.0
    nu = 0.3
    G = E / (2 * (1 + nu))
    A = 0.01
    I = 1e-06
    Iy = I
    Iz = I
    J = 2.0 * I
    I_rho = Iy + Iz
    L_total = 2.0
    n_el = 16
    L_e = L_total / n_el

    def ke_local(E, G, A, Iy, Iz, J, L):
        K = np.zeros((12, 12), dtype=float)
        kax = E * A / L
        K[0, 0] += kax
        K[0, 6] -= kax
        K[6, 0] -= kax
        K[6, 6] += kax
        kt = G * J / L
        K[3, 3] += kt
        K[3, 9] -= kt
        K[9, 3] -= kt
        K[9, 9] += kt
        L2 = L * L
        L3 = L2 * L
        c_z = E * Iz / L3
        kbz = c_z * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]])
        idx_bz = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                K[idx_bz[a], idx_bz[b]] += kbz[a, b]
        c_y = E * Iy / L3
        kby = c_y * np.array([[12.0, -6.0 * L, -12.0, -6.0 * L], [-6.0 * L, 4.0 * L2, 6.0 * L, 2.0 * L2], [-12.0, 6.0 * L, 12.0, 6.0 * L], [-6.0 * L, 2.0 * L2, 6.0 * L, 4.0 * L2]])
        idx_by = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                K[idx_by[a], idx_by[b]] += kby[a, b]
        return K
    n_nodes = n_el + 1
    ndof = 6 * n_nodes
    K_e_glob = np.zeros((ndof, ndof), dtype=float)
    for e in range(n_el):
        Ke = ke_local(E, G, A, Iy, Iz, J, L_e)
        i = e
        j = e + 1
        dof_idx = [6 * i + d for d in range(6)] + [6 * j + d for d in range(6)]
        for a in range(12):
            for b in range(12):
                K_e_glob[dof_idx[a], dof_idx[b]] += Ke[a, b]

    def Kg_global(P):
        Kg = np.zeros((ndof, ndof), dtype=float)
        for e in range(n_el):
            i = e
            j = e + 1
            dof_idx = [6 * i + d for d in range(6)] + [6 * j + d for d in range(6)]
            Kg_loc = fcn(L_e, A, I_rho, -P, 0.0, 0.0, 0.0, 0.0, 0.0)
            for a in range(12):
                for b in range(12):
                    Kg[dof_idx[a], dof_idx[b]] += Kg_loc[a, b]
        return Kg
    free = list(range(6, ndof))

    def min_eig(P):
        Kt = K_e_glob + Kg_global(P)
        Kt = 0.5 * (Kt + Kt.T)
        Kt_red = Kt[np.ix_(free, free)]
        vals = np.linalg.eigvalsh(Kt_red)
        return float(vals[0])
    P_cr_analytical = np.pi ** 2 * E * I / (4.0 * L_total ** 2)
    P_low = 0.0
    P_high = 2.0 * P_cr_analytical
    val_low = min_eig(P_low)
    val_high = min_eig(P_high)
    iters = 0
    while val_high > 0.0 and iters < 40:
        P_high *= 2.0
        val_high = min_eig(P_high)
        iters += 1
    assert val_low > 0.0
    assert val_high < 0.0
    for _ in range(60):
        P_mid = 0.5 * (P_low + P_high)
        v_mid = min_eig(P_mid)
        if v_mid > 0.0:
            P_low = P_mid
        else:
            P_high = P_mid
        if abs(P_high - P_low) / max(P_cr_analytical, 1.0) < 1e-06:
            break
    P_est = 0.5 * (P_low + P_high)
    rel_err = abs(P_est - P_cr_analytical) / P_cr_analytical
    assert rel_err < 0.05