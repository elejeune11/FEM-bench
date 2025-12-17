def MSA_3D_elastic_critical_load_CC1_H10_T3(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    import numpy as np
    import scipy
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an (N, 3) array of floats.')
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes

    def dof_indices_for_node(n):
        base = 6 * n
        return np.arange(base, base + 6, dtype=int)
    K = np.zeros((ndof, ndof), dtype=float)
    P = np.zeros(ndof, dtype=float)
    if nodal_loads is not None:
        for n, loads in nodal_loads.items():
            if n < 0 or n >= n_nodes:
                raise ValueError(f'Invalid node index in nodal_loads: {n}')
            loads = np.asarray(loads, dtype=float)
            if loads.shape != (6,):
                raise ValueError('Each load entry must be a 6-element iterable [Fx, Fy, Fz, Mx, My, Mz].')
            dofs = dof_indices_for_node(n)
            P[dofs] += loads
    fixed = np.zeros(ndof, dtype=bool)
    if boundary_conditions is not None:
        for n, bcs in boundary_conditions.items():
            if n < 0 or n >= n_nodes:
                raise ValueError(f'Invalid node index in boundary_conditions: {n}')
            bcs = np.asarray(bcs, dtype=int)
            if bcs.shape != (6,):
                raise ValueError('Each BC entry must be a 6-element iterable of 0 (free) or 1 (fixed).')
            fixed[dof_indices_for_node(n)] = bcs.astype(bool)
    free = ~fixed

    def compute_local_axes(ni, nj, local_z_vec):
        xi = node_coords[ni]
        xj = node_coords[nj]
        v = xj - xi
        L = float(np.linalg.norm(v))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element has zero or invalid length.')
        ex = v / L
        if local_z_vec is None:
            w = np.array([0.0, 0.0, 1.0], dtype=float)
            if np.linalg.norm(np.cross(ex, w)) < 1e-12:
                w = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            w = np.asarray(local_z_vec, dtype=float)
            if w.shape != (3,) or not np.all(np.isfinite(w)):
                raise ValueError('local_z must be a length-3 finite vector.')
            if np.linalg.norm(w) < 1e-12:
                raise ValueError('local_z must be non-zero.')
        ey_temp = np.cross(w, ex)
        n_ey = np.linalg.norm(ey_temp)
        if n_ey < 1e-12:
            alt = np.array([0.0, 1.0, 0.0], dtype=float) if np.linalg.norm(np.cross(ex, [0.0, 1.0, 0.0])) > 1e-08 else np.array([0.0, 0.0, 1.0], dtype=float)
            ey_temp = np.cross(alt, ex)
            n_ey = np.linalg.norm(ey_temp)
            if n_ey < 1e-12:
                raise ValueError('Failed to construct element local axes (degenerate orientation).')
        ey = ey_temp / n_ey
        ez = np.cross(ex, ey)
        R = np.vstack((ex, ey, ez))
        return (R, L)

    def element_local_elastic_stiffness(E, G, A, Iy, Iz, J, L):
        k = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        k[0, 0] += k_ax
        k[0, 6] += -k_ax
        k[6, 0] += -k_ax
        k[6, 6] += k_ax
        k_t = G * J / L
        k[3, 3] += k_t
        k[3, 9] += -k_t
        k[9, 3] += -k_t
        k[9, 9] += k_t
        EI = E * Iz
        L2 = L * L
        L3 = L2 * L
        kb = EI / L3 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float)
        idx = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                k[idx[a], idx[b]] += kb[a, b]
        EI = E * Iy
        kb = EI / L3 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float)
        idx = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                k[idx[a], idx[b]] += kb[a, b]
        return k

    def element_local_geometric_stiffness(N, L):
        Kg = np.zeros((12, 12), dtype=float)
        if abs(N) <= 0.0:
            return Kg
        L2 = L * L
        c = N / (30.0 * L)
        k4 = c * np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L2, -3.0 * L, -1.0 * L2], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -1.0 * L2, -3.0 * L, 4.0 * L2]], dtype=float)
        idx_y = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                Kg[idx_y[a], idx_y[b]] += k4[a, b]
        idx_z = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                Kg[idx_z[a], idx_z[b]] += k4[a, b]
        return Kg

    def transformation_matrix(R):
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        return T
    for e in elements:
        ni = int(e['node_i'])
        nj = int(e['node_j'])
        if ni < 0 or ni >= n_nodes or nj < 0 or (nj >= n_nodes):
            raise ValueError('Element node indices out of range.')
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['I_y'])
        Iz = float(e['I_z'])
        J = float(e['J'])
        local_z = e.get('local_z', None)
        R, L = compute_local_axes(ni, nj, local_z)
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element has zero or invalid length.')
        G = E / (2.0 * (1.0 + nu))
        Ke_local = element_local_elastic_stiffness(E, G, A, Iy, Iz, J, L)
        T = transformation_matrix(R)
        Ke_global = T.T @ Ke_local @ T
        dofs_i = dof_indices_for_node(ni)
        dofs_j = dof_indices_for_node(nj)
        edofs = np.hstack((dofs_i, dofs_j))
        K[np.ix_(edofs, edofs)] += Ke_global
    K_ff = K[np.ix_(free, free)]
    P_f = P[free]
    if K_ff.size == 0:
        raise ValueError('No free DOFs; boundary conditions overly constrained.')
    try:
        try:
            cond_est = np.linalg.cond(K_ff)
            if not np.isfinite(cond_est) or cond_est > 1000000000000.0:
                raise ValueError('Reduced stiffness matrix is singular or ill-conditioned.')
        except Exception:
            pass
        u_f = np.linalg.solve(K_ff, P_f)
    except np.linalg.LinAlgError as err:
        raise ValueError(f'Failed to solve Ku=P on free DOFs: {err}')
    u = np.zeros(ndof, dtype=float)
    u[free] = u_f
    Kg = np.zeros((ndof, ndof), dtype=float)
    any_nonzero_kg = False
    for e in elements:
        ni = int(e['node_i'])
        nj = int(e['node_j'])
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['I_y'])
        Iz = float(e['I_z'])
        J = float(e['J'])
        local_z = e.get('local_z', None)
        R, L = compute_local_axes(ni, nj, local_z)
        G = E / (2.0 * (1.0 + nu))
        Ke_local = element_local_elastic_stiffness(E, G, A, Iy, Iz, J, L)
        T = transformation_matrix(R)
        dofs_i = dof_indices_for_node(ni)
        dofs_j = dof_indices_for_node(nj)
        edofs = np.hstack((dofs_i, dofs_j))
        d_global_elem = u[edofs]
        d_local = T @ d_global_elem
        N = E * A / L * (d_local[6] - d_local[0])
        if abs(N) > 0.0:
            any_nonzero_kg = True
        Kg_local = element_local_geometric_stiffness(N, L)
        Kg_global = T.T @ Kg_local @ T
        Kg[np.ix_(edofs, edofs)] += Kg_global
    if not any_nonzero_kg or not np.any(np.abs(Kg) > 0.0):
        raise ValueError('Geometric stiffness matrix is zero; no reference stress state. Cannot perform buckling analysis.')
    K_ff = K[np.ix_(free, free)]
    Kg_ff = Kg[np.ix_(free, free)]
    B = -Kg_ff
    try:
        eigvals, eigvecs = scipy.linalg.eig(K_ff, B)
    except Exception as err:
        raise ValueError(f'Generalized eigenvalue problem failed: {err}')
    eigvals = np.asarray(eigvals)
    eigvecs = np.asarray(eigvecs)
    real_mask = np.abs(eigvals.imag) <= 1e-08 * np.maximum(1.0, np.abs(eigvals.real))
    pos_mask = eigvals.real > 1e-12
    sel = np.where(real_mask & pos_mask)[0]
    if sel.size == 0:
        try:
            scipy.linalg.cholesky(B, lower=True, overwrite_a=False, check_finite=True)
            try:
                eigvals2, eigvecs2 = scipy.linalg.eigh(K_ff, B)
                eigvals2 = np.asarray(eigvals2)
                pos = np.where(eigvals2 > 1e-12)[0]
                if pos.size == 0:
                    raise ValueError('No positive eigenvalues found in symmetric EVP.')
                idx = pos[np.argmin(eigvals2[pos])]
                lam = float(eigvals2[idx])
                vec_free = eigvecs2[:, idx]
            except Exception as err2:
                raise ValueError(f'Symmetric generalized EVP failed: {err2}')
        except Exception:
            raise ValueError('No positive real eigenvalue found for buckling.')
    else:
        vals_pos = eigvals.real[sel]
        idx_local = sel[np.argmin(vals_pos)]
        lam = float(eigvals.real[idx_local])
        vec_free = eigvecs[:, idx_local]
    mode = np.zeros(ndof, dtype=float)
    vec_free_real = np.real(vec_free)
    mode[free] = vec_free_real
    return (lam, mode)