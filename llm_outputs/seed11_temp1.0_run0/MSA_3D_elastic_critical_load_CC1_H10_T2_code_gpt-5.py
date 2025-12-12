def MSA_3D_elastic_critical_load_CC1_H10_T2(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    import numpy as np
    import scipy

    def _normalize(v):
        n = np.linalg.norm(v)
        if n == 0.0:
            raise ValueError('Zero-length vector encountered during local axis construction.')
        return v / n

    def _build_rotation(node_i, node_j, local_z_hint=None):
        xi = node_coords[node_i]
        xj = node_coords[node_j]
        ex = xj - xi
        L = np.linalg.norm(ex)
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element has zero or invalid length.')
        ex = ex / L
        if local_z_hint is None:
            ez_hint = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(ez_hint, ex)) > 0.999:
                ez_hint = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            ez_hint = np.array(local_z_hint, dtype=float)
            n = np.linalg.norm(ez_hint)
            if n == 0.0 or abs(np.dot(ez_hint / n, ex)) > 0.999:
                ez_hint = np.array([0.0, 0.0, 1.0], dtype=float)
                if abs(np.dot(ez_hint, ex)) > 0.999:
                    ez_hint = np.array([0.0, 1.0, 0.0], dtype=float)
        ez = ez_hint - np.dot(ez_hint, ex) * ex
        ez = _normalize(ez)
        ey = np.cross(ez, ex)
        ey = _normalize(ey)
        R = np.column_stack((ex, ey, ez))
        return (R, L)

    def _local_elastic_stiffness(E, nu, A, I_y, I_z, J, L):
        G = E / (2.0 * (1.0 + nu))
        k = np.zeros((12, 12), dtype=float)
        EA_L = E * A / L
        k[0, 0] += EA_L
        k[0, 6] += -EA_L
        k[6, 0] += -EA_L
        k[6, 6] += EA_L
        GJ_L = G * J / L
        k[3, 3] += GJ_L
        k[3, 9] += -GJ_L
        k[9, 3] += -GJ_L
        k[9, 9] += GJ_L
        s = E * I_z / L ** 3
        k[1, 1] += 12.0 * s
        k[1, 5] += 6.0 * L * s
        k[1, 7] += -12.0 * s
        k[1, 11] += 6.0 * L * s
        k[5, 1] += 6.0 * L * s
        k[5, 5] += 4.0 * L * L * s
        k[5, 7] += -6.0 * L * s
        k[5, 11] += 2.0 * L * L * s
        k[7, 1] += -12.0 * s
        k[7, 5] += -6.0 * L * s
        k[7, 7] += 12.0 * s
        k[7, 11] += -6.0 * L * s
        k[11, 1] += 6.0 * L * s
        k[11, 5] += 2.0 * L * L * s
        k[11, 7] += -6.0 * L * s
        k[11, 11] += 4.0 * L * L * s
        s = E * I_y / L ** 3
        k[2, 2] += 12.0 * s
        k[2, 4] += -6.0 * L * s
        k[2, 8] += -12.0 * s
        k[2, 10] += -6.0 * L * s
        k[4, 2] += -6.0 * L * s
        k[4, 4] += 4.0 * L * L * s
        k[4, 8] += 6.0 * L * s
        k[4, 10] += 2.0 * L * L * s
        k[8, 2] += -12.0 * s
        k[8, 4] += 6.0 * L * s
        k[8, 8] += 12.0 * s
        k[8, 10] += 6.0 * L * s
        k[10, 2] += -6.0 * L * s
        k[10, 4] += 2.0 * L * L * s
        k[10, 8] += 6.0 * L * s
        k[10, 10] += 4.0 * L * L * s
        return k

    def _T_from_R(R):
        T = np.zeros((12, 12), dtype=float)
        for n in range(2):
            T[n * 6 + 0:n * 6 + 3, n * 6 + 0:n * 6 + 3] = R
            T[n * 6 + 3:n * 6 + 6, n * 6 + 3:n * 6 + 6] = R
        return T
    n_nodes = node_coords.shape[0]
    dof_per_node = 6
    n_dof = n_nodes * dof_per_node
    K = np.zeros((n_dof, n_dof), dtype=float)
    P = np.zeros(n_dof, dtype=float)
    if nodal_loads:
        for (n, loads) in nodal_loads.items():
            arr = np.array(loads, dtype=float).reshape(-1)
            if arr.size != 6:
                raise ValueError('Each nodal load entry must have 6 components.')
            P[n * 6:(n + 1) * 6] += arr
    elem_cache = []
    for el in elements:
        i = int(el['node_i'])
        j = int(el['node_j'])
        E = float(el['E'])
        nu = float(el['nu'])
        A = float(el['A'])
        I_y = float(el['I_y'])
        I_z = float(el['I_z'])
        J = float(el['J'])
        local_z_hint = el.get('local_z', None)
        (R, L) = _build_rotation(i, j, local_z_hint)
        T = _T_from_R(R)
        k_loc = _local_elastic_stiffness(E, nu, A, I_y, I_z, J, L)
        k_glb = T.T @ k_loc @ T
        dofs = np.concatenate((np.arange(i * 6, i * 6 + 6, dtype=int), np.arange(j * 6, j * 6 + 6, dtype=int)))
        K[np.ix_(dofs, dofs)] += k_glb
        I_rho = I_y + I_z
        elem_cache.append((i, j, L, A, I_rho, T, k_loc, dofs))
    fixed = np.zeros(n_dof, dtype=bool)
    if boundary_conditions:
        for (n, bc) in boundary_conditions.items():
            arr = np.array(bc, dtype=int).reshape(-1)
            if arr.size != 6:
                raise ValueError('Each boundary condition entry must have 6 components.')
            f = arr.astype(bool)
            fixed[n * 6:(n + 1) * 6] = f
    free = ~fixed
    if not np.any(free):
        raise ValueError('All DOFs are constrained; no free DOFs to analyze.')
    K_ff = K[np.ix_(free, free)]
    P_f = P[free]
    try:
        condK = np.linalg.cond(K_ff)
        if not np.isfinite(condK) or condK > 1000000000000.0:
            raise ValueError('Reduced elastic stiffness matrix is ill-conditioned or singular.')
        u_f = scipy.linalg.solve(K_ff, P_f, assume_a='sym')
    except Exception as e:
        raise ValueError(f'Failed to solve static equilibrium: {e}')
    u = np.zeros(n_dof, dtype=float)
    u[free] = u_f
    K_g = np.zeros((n_dof, n_dof), dtype=float)
    for (i, j, L, A, I_rho, T, k_loc, dofs) in elem_cache:
        u_e = u[dofs]
        u_loc = T @ u_e
        f_loc = k_loc @ u_loc
        Fx2 = float(f_loc[6])
        Mx2 = float(f_loc[9])
        My1 = float(f_loc[4])
        Mz1 = float(f_loc[5])
        My2 = float(f_loc[10])
        Mz2 = float(f_loc[11])
        k_g_loc = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_glb = T.T @ k_g_loc @ T
        K_g[np.ix_(dofs, dofs)] += k_g_glb
    K_g_ff = K_g[np.ix_(free, free)]
    B = -K_g_ff
    try:
        if not np.all(np.isfinite(B)):
            raise ValueError('Geometric stiffness contains invalid entries.')
        (evals, evecs) = scipy.linalg.eig(K_ff, B)
    except Exception as e:
        raise ValueError(f'Failed to solve generalized eigenproblem: {e}')
    evals = np.asarray(evals)
    evecs = np.asarray(evecs)
    imag_tol = 1e-06
    re = evals.real
    im = evals.imag
    mask_real = np.abs(im) <= imag_tol * np.maximum(1.0, np.abs(re))
    mask_pos = re > 0.0
    mask = mask_real & mask_pos
    if not np.any(mask):
        raise ValueError('No positive real eigenvalue found for buckling problem.')
    idxs = np.where(mask)[0]
    min_idx = idxs[np.argmin(re[idxs])]
    lam = float(re[min_idx])
    mode_f = evecs[:, min_idx].real
    mode = np.zeros(n_dof, dtype=float)
    mode[free] = mode_f
    return (lam, mode)