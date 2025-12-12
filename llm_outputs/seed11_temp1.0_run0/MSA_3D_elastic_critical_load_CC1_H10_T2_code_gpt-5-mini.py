def MSA_3D_elastic_critical_load_CC1_H10_T2(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    """
    Perform linear (eigenvalue) buckling analysis for a 3D frame and return the
    elastic critical load factor and associated global buckling mode shape.
    (Docstring unchanged from prompt)
    """
    node_coords = np.asarray(node_coords, dtype=float)
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    P = np.zeros(ndof, dtype=float)
    for ni in range(n_nodes):
        loads = nodal_loads.get(ni, None)
        if loads is None:
            continue
        arr = np.asarray(loads, dtype=float)
        if arr.size != 6:
            raise ValueError('Each nodal load must have 6 components')
        P[6 * ni:6 * ni + 6] = P[6 * ni:6 * ni + 6] + arr

    def local_elastic_stiffness_3d_beam(L, E, G, A, I_y, I_z, J):
        k = np.zeros((12, 12), dtype=float)
        k[0, 0] = k[6, 6] = E * A / L
        k[0, 6] = k[6, 0] = -E * A / L
        k[3, 3] = k[9, 9] = G * J / L
        k[3, 9] = k[9, 3] = -G * J / L
        kzz = E * I_z / L ** 3
        mat_z = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L]], dtype=float) * (E * I_z / L ** 3) / (E * I_z / L ** 3)
        mat_z = mat_z * (E * I_z / L ** 3)
        idx_z = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                k[idx_z[a], idx_z[b]] = mat_z[a, b]
        mat_y = np.array([[12.0, -6.0 * L, -12.0, -6.0 * L], [-6.0 * L, 4.0 * L * L, 6.0 * L, 2.0 * L * L], [-12.0, 6.0 * L, 12.0, 6.0 * L], [-6.0 * L, 2.0 * L * L, 6.0 * L, 4.0 * L * L]], dtype=float) * (E * I_y / L ** 3) / (E * I_y / L ** 3)
        mat_y = mat_y * (E * I_y / L ** 3)
        idx_y = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                k[idx_y[a], idx_y[b]] = mat_y[a, b]
        return k
    for elem in elements:
        ni = int(elem['node_i'])
        nj = int(elem['node_j'])
        xi = node_coords[ni]
        xj = node_coords[nj]
        Lvec = xj - xi
        L = np.linalg.norm(Lvec)
        if L <= 0.0:
            raise ValueError('Element with zero length')
        ex = Lvec / L
        local_z = elem.get('local_z', None)
        if local_z is None:
            gz = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(gz, ex)) > 0.999999:
                gz = np.array([0.0, 1.0, 0.0], dtype=float)
            ez = gz
        else:
            ez = np.asarray(local_z, dtype=float)
            if ez.shape != (3,):
                raise ValueError('local_z must be length 3')
            if np.linalg.norm(np.cross(ez, ex)) < 1e-08:
                gz = np.array([0.0, 0.0, 1.0], dtype=float)
                if abs(np.dot(gz, ex)) > 0.999999:
                    gz = np.array([0.0, 1.0, 0.0], dtype=float)
                ez = gz
        ey = np.cross(ez, ex)
        if np.linalg.norm(ey) < 1e-12:
            raise ValueError('local_z parallel to element axis')
        ey = ey / np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        ez = ez / np.linalg.norm(ez)
        R = np.zeros((3, 3), dtype=float)
        R[:, 0] = ex
        R[:, 1] = ey
        R[:, 2] = ez
        T = np.zeros((12, 12), dtype=float)
        for node_block in (0, 6):
            T[node_block:node_block + 3, node_block:node_block + 3] = R
            T[node_block + 3:node_block + 6, node_block + 3:node_block + 6] = R
        E = float(elem['E'])
        nu = float(elem['nu'])
        A = float(elem['A'])
        I_y = float(elem['I_y'])
        I_z = float(elem['I_z'])
        J = float(elem['J'])
        G = E / (2.0 * (1.0 + nu))
        k_e_loc = local_elastic_stiffness_3d_beam(L, E, G, A, I_y, I_z, J)
        k_e_glob = T @ k_e_loc @ T.T
        dofs_i = np.arange(6 * ni, 6 * ni + 6)
        dofs_j = np.arange(6 * nj, 6 * nj + 6)
        edofs = np.hstack((dofs_i, dofs_j))
        for a in range(12):
            ia = edofs[a]
            for b in range(12):
                ib = edofs[b]
                K[ia, ib] += k_e_glob[a, b]
    fixed = np.zeros(ndof, dtype=bool)
    for (n, bc) in boundary_conditions.items():
        arr = np.asarray(bc, dtype=int)
        if arr.size != 6:
            raise ValueError('Each BC entry must have 6 entries of 0 or 1')
        for j in range(6):
            if arr[j] == 1:
                fixed[n * 6 + j] = True
    free = ~fixed
    free_idx = np.nonzero(free)[0]
    fixed_idx = np.nonzero(fixed)[0]
    if free_idx.size == 0:
        raise ValueError('No free degrees of freedom')
    K_ff = K[np.ix_(free_idx, free_idx)]
    P_f = P[free_idx]
    try:
        u_f = np.linalg.solve(K_ff, P_f)
    except np.linalg.LinAlgError as e:
        raise ValueError('Singular stiffness matrix for the applied BCs / loads') from e
    u = np.zeros(ndof, dtype=float)
    u[free_idx] = u_f
    u[fixed_idx] = 0.0
    K_g = np.zeros((ndof, ndof), dtype=float)
    for elem in elements:
        ni = int(elem['node_i'])
        nj = int(elem['node_j'])
        xi = node_coords[ni]
        xj = node_coords[nj]
        Lvec = xj - xi
        L = np.linalg.norm(Lvec)
        ex = Lvec / L
        local_z = elem.get('local_z', None)
        if local_z is None:
            gz = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(gz, ex)) > 0.999999:
                gz = np.array([0.0, 1.0, 0.0], dtype=float)
            ez = gz
        else:
            ez = np.asarray(local_z, dtype=float)
            if np.linalg.norm(np.cross(ez, ex)) < 1e-08:
                gz = np.array([0.0, 0.0, 1.0], dtype=float)
                if abs(np.dot(gz, ex)) > 0.999999:
                    gz = np.array([0.0, 1.0, 0.0], dtype=float)
                ez = gz
        ey = np.cross(ez, ex)
        if np.linalg.norm(ey) < 1e-12:
            raise ValueError('local_z parallel to element axis')
        ey = ey / np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        ez = ez / np.linalg.norm(ez)
        R = np.zeros((3, 3), dtype=float)
        R[:, 0] = ex
        R[:, 1] = ey
        R[:, 2] = ez
        T = np.zeros((12, 12), dtype=float)
        for node_block in (0, 6):
            T[node_block:node_block + 3, node_block:node_block + 3] = R
            T[node_block + 3:node_block + 6, node_block + 3:node_block + 6] = R
        E = float(elem['E'])
        nu = float(elem['nu'])
        A = float(elem['A'])
        I_y = float(elem['I_y'])
        I_z = float(elem['I_z'])
        J = float(elem['J'])
        G = E / (2.0 * (1.0 + nu))
        k_e_loc = local_elastic_stiffness_3d_beam(L, E, G, A, I_y, I_z, J)
        dofs_i = np.arange(6 * ni, 6 * ni + 6)
        dofs_j = np.arange(6 * nj, 6 * nj + 6)
        edofs = np.hstack((dofs_i, dofs_j))
        u_e_global = u[edofs]
        u_e_local = T.T @ u_e_global
        f_int_local = k_e_loc @ u_e_local
        f1 = f_int_local[0:6]
        f2 = f_int_local[6:12]
        Fx2 = float(f2[0])
        Mx2 = float(f2[3])
        My1 = float(f1[4])
        Mz1 = float(f1[5])
        My2 = float(f2[4])
        Mz2 = float(f2[5])
        k_g_loc = local_geometric_stiffness_matrix_3D_beam(L, A, J, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_glob = T @ k_g_loc @ T.T
        for a in range(12):
            ia = edofs[a]
            for b in range(12):
                ib = edofs[b]
                K_g[ia, ib] += k_g_glob[a, b]
    K_ff = K[np.ix_(free_idx, free_idx)]
    Kg_ff = K_g[np.ix_(free_idx, free_idx)]
    B = -Kg_ff
    A = K_ff
    try:
        (eigvals, eigvecs) = scipy.linalg.eig(A, B)
    except Exception as e:
        raise ValueError('Generalized eigenproblem failed') from e
    eigvals = np.asarray(eigvals)
    eigvecs = np.asarray(eigvecs)
    imag_tol = 1e-08
    real_mask = np.abs(np.imag(eigvals)) < imag_tol
    if not np.any(real_mask):
        raise ValueError('No sufficiently real eigenvalues found')
    real_eigvals = np.real(eigvals[real_mask])
    real_eigvecs = eigvecs[:, real_mask]
    pos_mask = real_eigvals > 1e-12
    if not np.any(pos_mask):
        raise ValueError('No positive eigenvalue found')
    pos_vals = real_eigvals[pos_mask]
    pos_vecs = real_eigvecs[:, pos_mask]
    idx_min = np.argmin(pos_vals)
    lambda_min = float(pos_vals[idx_min])
    phi_free = np.real(pos_vecs[:, idx_min])
    phi_full = np.zeros(ndof, dtype=float)
    phi_full[free_idx] = phi_free
    phi_full[fixed_idx] = 0.0
    return (lambda_min, phi_full)