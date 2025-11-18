def elastic_critical_load_analysis_frame_3D_part_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    import numpy as np
    import scipy
    n_nodes = int(node_coords.shape[0])
    if node_coords.shape[1] != 3:
        raise ValueError('node_coords must have shape (n_nodes, 3).')
    ndof = 6 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    P = np.zeros(ndof, dtype=float)
    for (n, load) in (nodal_loads or {}).items():
        if not 0 <= n < n_nodes:
            raise ValueError(f'Load specified for invalid node index {n}.')
        load = np.asarray(load, dtype=float).reshape(-1)
        if load.shape[0] != 6:
            raise ValueError(f'Load vector for node {n} must have length 6.')
        base = 6 * n
        P[base:base + 6] += load

    def _element_transformation(i_node: int, j_node: int, local_z):
        xi = np.asarray(node_coords[i_node], dtype=float)
        xj = np.asarray(node_coords[j_node], dtype=float)
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element length must be positive.')
        ex = dx / L
        tol = 1e-12
        if local_z is not None:
            z_hint = np.asarray(local_z, dtype=float).reshape(-1)
            if z_hint.shape[0] != 3:
                raise ValueError('local_z must be length-3 vector or None.')
            nz = np.linalg.norm(z_hint)
            if nz <= tol:
                raise ValueError('Provided local_z vector has near-zero magnitude.')
            z_hint = z_hint / nz
            if np.linalg.norm(np.cross(z_hint, ex)) <= 1e-08:
                z_hint = np.array([0.0, 0.0, 1.0])
                if np.linalg.norm(np.cross(z_hint, ex)) <= 1e-08:
                    z_hint = np.array([0.0, 1.0, 0.0])
        else:
            z_hint = np.array([0.0, 0.0, 1.0])
            if np.linalg.norm(np.cross(z_hint, ex)) <= 1e-08:
                z_hint = np.array([0.0, 1.0, 0.0])
        ey = np.cross(z_hint, ex)
        ny = np.linalg.norm(ey)
        if ny <= tol:
            alt = np.array([1.0, 0.0, 0.0])
            if np.linalg.norm(np.cross(alt, ex)) <= 1e-08:
                alt = np.array([0.0, 1.0, 0.0])
            ey = np.cross(alt, ex)
            ny = np.linalg.norm(ey)
            if ny <= tol:
                raise ValueError('Failed to determine a valid local triad.')
        ey = ey / ny
        ez = np.cross(ex, ey)
        ez = ez / np.linalg.norm(ez)
        R = np.column_stack((ex, ey, ez))
        T = np.zeros((12, 12), dtype=float)
        Rt = R.T
        T[0:3, 0:3] = Rt
        T[3:6, 3:6] = Rt
        T[6:9, 6:9] = Rt
        T[9:12, 9:12] = Rt
        return (L, T)
    for e in elements:
        i = int(e['node_i'])
        j = int(e['node_j'])
        if not (0 <= i < n_nodes and 0 <= j < n_nodes):
            raise ValueError('Element connectivity indices out of range.')
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['I_y'])
        Iz = float(e['I_z'])
        J = float(e['J'])
        local_z = e.get('local_z', None)
        (L, T) = _element_transformation(i, j, local_z)
        k_loc = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        k_gl = T.T @ k_loc @ T
        edofs = np.r_[np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)]
        K[np.ix_(edofs, edofs)] += k_gl
    K = 0.5 * (K + K.T)
    constrained = np.zeros(ndof, dtype=bool)
    if boundary_conditions is not None:
        for (n, spec) in boundary_conditions.items():
            if not 0 <= n < n_nodes:
                raise ValueError(f'Boundary condition for invalid node index {n}.')
            base = 6 * int(n)
            spec_list = list(spec)
            use_bool = len(spec_list) == 6 and all((type(x) is bool or isinstance(x, (np.bool_,)) for x in spec_list))
            if use_bool:
                for k in range(6):
                    if bool(spec_list[k]):
                        constrained[base + k] = True
            else:
                for idx in spec_list:
                    ii = int(idx)
                    if not 0 <= ii < 6:
                        raise ValueError(f'Invalid DOF index {ii} in boundary condition for node {n}.')
                    constrained[base + ii] = True
    free = np.nonzero(~constrained)[0]
    if free.size == 0:
        raise ValueError('No free DOFs remain after applying boundary conditions.')
    K_ff = K[np.ix_(free, free)]
    P_f = P[free]
    try:
        u_f = scipy.linalg.solve(K_ff, P_f, assume_a='sym')
    except Exception as err:
        raise ValueError(f'Failed to solve static system K u = P on free DOFs: {err}')
    u = np.zeros(ndof, dtype=float)
    u[free] = u_f
    K_g = np.zeros((ndof, ndof), dtype=float)
    for e in elements:
        i = int(e['node_i'])
        j = int(e['node_j'])
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['I_y'])
        Iz = float(e['I_z'])
        J = float(e['J'])
        I_rho = float(e['I_rho'])
        local_z = e.get('local_z', None)
        (L, T) = _element_transformation(i, j, local_z)
        k_loc = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        edofs = np.r_[np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)]
        u_e_gl = u[edofs]
        u_e_loc = T @ u_e_gl
        f_e_loc = k_loc @ u_e_loc
        Fx2 = float(f_e_loc[6])
        Mx2 = float(f_e_loc[9])
        My1 = float(f_e_loc[4])
        Mz1 = float(f_e_loc[5])
        My2 = float(f_e_loc[10])
        Mz2 = float(f_e_loc[11])
        k_g_loc = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_gl = T.T @ k_g_loc @ T
        K_g[np.ix_(edofs, edofs)] += k_g_gl
    K_g = 0.5 * (K_g + K_g.T)
    K_g_ff = K_g[np.ix_(free, free)]
    try:
        (eigvals, eigvecs) = scipy.linalg.eigh(-K_g_ff, K_ff)
    except Exception as err:
        raise ValueError(f'Failed to solve generalized eigenproblem: {err}')
    if eigvals.ndim != 1 or eigvecs.ndim != 2 or eigvecs.shape[0] != eigvals.shape[0]:
        raise ValueError('Eigen-solver returned invalid shapes.')
    finite_mask = np.isfinite(eigvals)
    vals = eigvals[finite_mask]
    vecs = eigvecs[:, finite_mask]
    if vals.size == 0:
        raise ValueError('No finite eigenvalues found.')
    tol_pos = 1e-10
    pos_mask = vals > tol_pos
    if not np.any(pos_mask):
        raise ValueError('No positive eigenvalue found (buckling not predicted under given load/state).')
    vals_pos = vals[pos_mask]
    vecs_pos = vecs[:, pos_mask]
    min_idx = int(np.argmin(vals_pos))
    lambda_cr = float(vals_pos[min_idx])
    phi_f = vecs_pos[:, min_idx]
    phi = np.zeros(ndof, dtype=float)
    phi[free] = phi_f
    return (lambda_cr, phi)