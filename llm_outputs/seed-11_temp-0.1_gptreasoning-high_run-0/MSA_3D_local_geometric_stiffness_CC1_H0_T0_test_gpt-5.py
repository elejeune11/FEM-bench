def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 3.2
    A = 0.045
    I_rho = 0.0023
    K = fcn(L, A, I_rho, Fx2=1.2, Mx2=0.3, My1=-1.1, Mz1=2.2, My2=0.7, Mz2=-0.5)
    assert isinstance(K, np.ndarray) and K.shape == (12, 12)
    assert np.allclose(K, K.T, atol=1e-12, rtol=0)
    K0 = fcn(L, A, I_rho, Fx2=0.0, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    assert np.allclose(K0, np.zeros((12, 12)), atol=1e-14, rtol=0)
    P = 10.0
    K_tension = fcn(L, A, I_rho, Fx2=+P, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    K_compression = fcn(L, A, I_rho, Fx2=-P, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    idx_z = [1, 5, 7, 11]
    idx_y = [2, 4, 8, 10]
    tr_tension_z = np.trace(K_tension[np.ix_(idx_z, idx_z)])
    tr_compress_z = np.trace(K_compression[np.ix_(idx_z, idx_z)])
    tr_tension_y = np.trace(K_tension[np.ix_(idx_y, idx_y)])
    tr_compress_y = np.trace(K_compression[np.ix_(idx_y, idx_y)])
    assert tr_tension_z > 0 and tr_compress_z < 0
    assert tr_tension_y > 0 and tr_compress_y < 0
    Kg_base = fcn(L, A, I_rho, Fx2=P, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    Kg_torsion = fcn(L, A, I_rho, Fx2=P, Mx2=5.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    Kg_bending = fcn(L, A, I_rho, Fx2=P, Mx2=0.0, My1=3.0, Mz1=-2.0, My2=-1.0, Mz2=4.0)
    assert not np.allclose(Kg_torsion, Kg_base, atol=1e-12, rtol=0)
    assert not np.allclose(Kg_bending, Kg_base, atol=1e-12, rtol=0)
    alpha = 2.75
    Kg1 = fcn(L, A, I_rho, Fx2=1.0, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    Kg2 = fcn(L, A, I_rho, Fx2=alpha, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    assert np.allclose(Kg2, alpha * Kg1, atol=1e-12, rtol=1e-12)

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads
    to the correct buckling load for a cantilever column. Compare the numerical
    result from an eigenvalue buckling analysis against the analytical Euler
    buckling load, allowing tolerance for discretization error.
    """
    E = 210000000000.0
    nu = 0.3
    G = E / (2 * (1 + nu))
    L_total = 2.0
    A = 0.01
    Iz = 8e-06
    Iy = 1.6e-05
    J = Iy + Iz
    I_rho = J
    n_el = 12
    L_e = L_total / n_el
    n_nodes = n_el + 1
    ndof_total = 6 * n_nodes
    Ke_global = np.zeros((ndof_total, ndof_total))
    Kg_global_ref = np.zeros((ndof_total, ndof_total))

    def ke_3d_eb(E, G, A, Iy, Iz, J, L):
        K = np.zeros((12, 12))
        k_ax = E * A / L
        K[0, 0] += k_ax
        K[0, 6] -= k_ax
        K[6, 0] -= k_ax
        K[6, 6] += k_ax
        k_t = G * J / L
        K[3, 3] += k_t
        K[3, 9] -= k_t
        K[9, 3] -= k_t
        K[9, 9] += k_t
        a = E * Iz / L ** 3
        mat_z = np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L * L, -6 * L, 2 * L * L], [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L * L, -6 * L, 4 * L * L]], dtype=float)
        idx_z = [1, 5, 7, 11]
        K[np.ix_(idx_z, idx_z)] += a * mat_z
        b = E * Iy / L ** 3
        mat_y = np.array([[12, -6 * L, -12, -6 * L], [-6 * L, 4 * L * L, 6 * L, 2 * L * L], [-12, 6 * L, 12, 6 * L], [-6 * L, 2 * L * L, 6 * L, 4 * L * L]], dtype=float)
        idx_y = [2, 4, 8, 10]
        K[np.ix_(idx_y, idx_y)] += b * mat_y
        return K
    for e in range(n_el):
        ke = ke_3d_eb(E, G, A, Iy, Iz, J, L_e)
        kg = fcn(L_e, A, I_rho, Fx2=-1.0, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
        dofs = np.r_[np.arange(6 * e, 6 * e + 6), np.arange(6 * (e + 1), 6 * (e + 1) + 6)]
        Ke_global[np.ix_(dofs, dofs)] += ke
        Kg_global_ref[np.ix_(dofs, dofs)] += kg
    bending_free_dofs = []
    for k in range(1, n_nodes):
        base = 6 * k
        bending_free_dofs.extend([base + 1, base + 2, base + 4, base + 5])
    Ke_ff = Ke_global[np.ix_(bending_free_dofs, bending_free_dofs)]
    Kg_ff_ref = Kg_global_ref[np.ix_(bending_free_dofs, bending_free_dofs)]
    H = -Kg_ff_ref
    eps = 1e-12 * np.trace(Ke_ff) / Ke_ff.shape[0]
    try:
        Lc = np.linalg.cholesky(H + eps * np.eye(H.shape[0]))
    except np.linalg.LinAlgError:
        Lc = np.linalg.cholesky(H + 100 * eps * np.eye(H.shape[0]))
    Y = np.linalg.solve(Lc, Ke_ff)
    C = np.linalg.solve(Lc.T, Y.T).T
    evals = np.linalg.eigvalsh(C)
    evals_pos = evals[evals > 1e-10]
    assert evals_pos.size > 0
    P_cr_num = evals_pos.min()
    P_cr_analytical = np.pi ** 2 * E * Iz / (2.0 * L_total) ** 2
    rel_err = abs(P_cr_num - P_cr_analytical) / P_cr_analytical
    assert rel_err < 0.08