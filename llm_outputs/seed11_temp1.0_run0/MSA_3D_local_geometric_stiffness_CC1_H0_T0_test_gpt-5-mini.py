def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 2.0
    A = 0.01
    I_rho = 1e-06
    K0 = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert isinstance(K0, np.ndarray)
    assert K0.shape == (12, 12)
    assert np.allclose(K0, K0.T, atol=1e-12)
    assert np.allclose(K0, np.zeros((12, 12)), atol=1e-12)
    Fx_tension = 1000.0
    Fx_compression = -1000.0
    K_tension = fcn(L, A, I_rho, Fx_tension, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_compression = fcn(L, A, I_rho, Fx_compression, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert not np.allclose(K_tension, 0.0)
    assert not np.allclose(K_compression, 0.0)
    lateral_indices = [1, 2, 4, 5, 7, 8, 10, 11]

    def lateral_norm(K):
        block = K[np.ix_(lateral_indices, lateral_indices)]
        return np.linalg.norm(block, ord='fro')
    norm_tension = lateral_norm(K_tension)
    norm_compression = lateral_norm(K_compression)
    assert norm_tension > norm_compression
    K_base = fcn(L, A, I_rho, 500.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_with_Mx = fcn(L, A, I_rho, 500.0, 1000.0, 0.0, 0.0, 0.0, 0.0)
    K_with_bending = fcn(L, A, I_rho, 500.0, 0.0, 1000.0, 2000.0, -1000.0, 500.0)
    assert not np.allclose(K_base, K_with_Mx)
    assert not np.allclose(K_base, K_with_bending)
    Fx_single = 200.0
    k = 5.0
    K_single = fcn(L, A, I_rho, Fx_single, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_scaled = fcn(L, A, I_rho, Fx_single * k, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(K_scaled, K_single * k, atol=1e-09, rtol=1e-06)

def test_euler_buckling_cantilever_column(fcn):
    """Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Tolerances allow for small discretization/numerical error.
    """
    L = 2.0
    E = 210000000000.0
    G = 80000000000.0
    A = 0.0001
    I = 1e-06
    J = 2e-06
    I_rho = I
    Ke = np.zeros((12, 12))
    Ke[0, 0] = Ke[6, 6] = E * A / L
    Ke[0, 6] = Ke[6, 0] = -E * A / L
    Ke[3, 3] = Ke[9, 9] = G * J / L
    Ke[3, 9] = Ke[9, 3] = -G * J / L
    idx_v = [1, 5, 7, 11]
    EI_z = E * I
    k_bz = np.array([[12 * EI_z / L ** 3, 6 * EI_z / L ** 2, -12 * EI_z / L ** 3, 6 * EI_z / L ** 2], [6 * EI_z / L ** 2, 4 * EI_z / L, -6 * EI_z / L ** 2, 2 * EI_z / L], [-12 * EI_z / L ** 3, -6 * EI_z / L ** 2, 12 * EI_z / L ** 3, -6 * EI_z / L ** 2], [6 * EI_z / L ** 2, 2 * EI_z / L, -6 * EI_z / L ** 2, 4 * EI_z / L]])
    for (i_row, r) in enumerate(idx_v):
        for (i_col, c) in enumerate(idx_v):
            Ke[r, c] += k_bz[i_row, i_col]
    idx_w = [2, 4, 8, 10]
    EI_y = E * I
    k_by = np.array([[12 * EI_y / L ** 3, -6 * EI_y / L ** 2, -12 * EI_y / L ** 3, -6 * EI_y / L ** 2], [-6 * EI_y / L ** 2, 4 * EI_y / L, 6 * EI_y / L ** 2, 2 * EI_y / L], [-12 * EI_y / L ** 3, 6 * EI_y / L ** 2, 12 * EI_y / L ** 3, 6 * EI_y / L ** 2], [-6 * EI_y / L ** 2, 2 * EI_y / L, 6 * EI_y / L ** 2, 4 * EI_y / L]])
    for (i_row, r) in enumerate(idx_w):
        for (i_col, c) in enumerate(idx_w):
            Ke[r, c] += k_by[i_row, i_col]
    free_dofs = list(range(6, 12))
    K_g_unit = fcn(L, A, I_rho, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def reduce_matrix(M):
        return M[np.ix_(free_dofs, free_dofs)]
    Ke_r = reduce_matrix(Ke)
    Kg_unit_r = reduce_matrix(K_g_unit)

    def min_eigenvalue(P):
        Ktot = Ke_r + P * Kg_unit_r
        vals = np.linalg.eigvals(Ktot)
        vals_real = np.real_if_close(vals, tol=1000000.0)
        vals_real = np.real(vals_real)
        return np.min(vals_real)
    P_euler = pi ** 2 * E * I / (4.0 * L ** 2)
    a = 0.0
    fa = min_eigenvalue(a)
    assert fa > 0.0
    b = P_euler * 2.0
    fb = min_eigenvalue(b)
    max_expand = 20
    expand_count = 0
    while fb > 0 and expand_count < max_expand:
        b *= 2.0
        fb = min_eigenvalue(b)
        expand_count += 1
    assert fb < 0.0
    tol = 1e-06 * P_euler
    for _ in range(60):
        m = 0.5 * (a + b)
        fm = min_eigenvalue(m)
        if abs(fm) < 1e-10:
            P_found = m
            break
        if fm > 0:
            a = m
        else:
            b = m
    else:
        P_found = 0.5 * (a + b)
    rel_error = abs(P_found - P_euler) / P_euler
    assert rel_error < 0.05