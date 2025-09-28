def elastic_critical_load_analysis_frame_3D_part_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = len(node_coords)
    n_dofs = 6 * n_nodes
    K = np.zeros((n_dofs, n_dofs))
    K_g = np.zeros((n_dofs, n_dofs))
    P = np.zeros(n_dofs)
    for (node_idx, loads) in nodal_loads.items():
        dof_indices = slice(6 * node_idx, 6 * node_idx + 6)
        P[dof_indices] = loads
    constrained_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        if isinstance(bc_spec[0], bool):
            constrained_dofs.update((6 * node_idx + i for (i, fixed) in enumerate(bc_spec) if fixed))
        else:
            constrained_dofs.update((6 * node_idx + i for i in bc_spec))
    free_dofs = list(set(range(n_dofs)) - constrained_dofs)
    for elem in elements:
        (node_i, node_j) = (elem['node_i'], elem['node_j'])
        xi = node_coords[node_i]
        xj = node_coords[node_j]
        dx = xj - xi
        L = np.linalg.norm(dx)
        ex = dx / L
        if elem.get('local_z') is not None:
            ez_temp = np.array(elem['local_z'])
            ey = np.cross(ez_temp, ex)
            ey = ey / np.linalg.norm(ey)
            ez = np.cross(ex, ey)
        else:
            if abs(ex[2]) < 0.99:
                ez_temp = np.array([0.0, 0.0, 1.0])
            else:
                ez_temp = np.array([0.0, 1.0, 0.0])
            ey = np.cross(ez_temp, ex)
            ey = ey / np.linalg.norm(ey)
            ez = np.cross(ex, ey)
        T = np.zeros((12, 12))
        R = np.vstack((ex, ey, ez))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        k_local = local_elastic_stiffness_matrix_3D_beam(elem['E'], elem['nu'], elem['A'], L, elem['Iy'], elem['Iz'], elem['J'])
        k_global = T.T @ k_local @ T
        dofs_i = slice(6 * node_i, 6 * node_i + 6)
        dofs_j = slice(6 * node_j, 6 * node_j + 6)
        dofs = np.r_[dofs_i, dofs_j]
        K[np.ix_(dofs, dofs)] += k_global
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(n_dofs)
    u[free_dofs] = u_f
    for elem in elements:
        (node_i, node_j) = (elem['node_i'], elem['node_j'])
        xi = node_coords[node_i]
        xj = node_coords[node_j]
        dx = xj - xi
        L = np.linalg.norm(dx)
        ex = dx / L
        if elem.get('local_z') is not None:
            ez_temp = np.array(elem['local_z'])
            ey = np.cross(ez_temp, ex)
            ey = ey / np.linalg.norm(ey)
            ez = np.cross(ex, ey)
        else:
            if abs(ex[2]) < 0.99:
                ez_temp = np.array([0.0, 0.0, 1.0])
            else:
                ez_temp = np.array([0.0, 1.0, 0.0])
            ey = np.cross(ez_temp, ex)
            ey = ey / np.linalg.norm(ey)
            ez = np.cross(ex, ey)
        T = np.zeros((12, 12))
        R = np.vstack((ex, ey, ez))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        u_elem = np.concatenate([u[6 * node_i:6 * node_i + 6], u[6 * node_j:6 * node_j + 6]])
        u_local = T @ u_elem
        f_local = k_local @ u_local
        Fx2 = f_local[6]
        Mx2 = f_local[9]
        (My1, Mz1) = (f_local[4], f_local[5])
        (My2, Mz2) = (f_local[10], f_local[11])
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, elem['A'], elem['I_rho'], Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = T.T @ k_g_local @ T
        dofs = np.r_[6 * node_i:6 * node_i + 6, 6 * node_j:6 * node_j + 6]
        K_g[np.ix_(dofs, dofs)] += k_g_global
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_g_ff = K_g[np.ix_(free_dofs, free_dofs)]
    (eigvals, eigvecs) = scipy.linalg.eigh(K_ff, -K_g_ff)
    pos_eigvals = eigvals[eigvals > 0]
    if len(pos_eigvals) == 0:
        raise ValueError('No positive eigenvalues found')
    critical_load_factor = np.min(pos_eigvals)
    mode_index = np.where(eigvals == critical_load_factor)[0][0]
    mode_shape = np.zeros(n_dofs)
    mode_shape[free_dofs] = eigvecs[:, mode_index]
    return (critical_load_factor, mode_shape)