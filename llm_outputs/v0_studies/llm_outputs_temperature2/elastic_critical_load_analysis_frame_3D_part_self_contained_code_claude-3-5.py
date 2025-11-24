def elastic_critical_load_analysis_frame_3D_part_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    K_g = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    for (node_idx, loads) in nodal_loads.items():
        P[6 * node_idx:6 * node_idx + 6] = loads
    constrained_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        if isinstance(bc_spec[0], bool):
            constrained_dofs.update((6 * node_idx + i for (i, fixed) in enumerate(bc_spec) if fixed))
        else:
            constrained_dofs.update((6 * node_idx + i for i in bc_spec))
    free_dofs = list(set(range(n_dof)) - constrained_dofs)
    for elem in elements:
        (node_i, node_j) = (elem['node_i'], elem['node_j'])
        xi = node_coords[node_i]
        xj = node_coords[node_j]
        dx = xj - xi
        L = np.linalg.norm(dx)
        ex = dx / L
        if elem['local_z'] is not None:
            ez_temp = np.array(elem['local_z'])
            ey = np.cross(ez_temp, ex)
            ey = ey / np.linalg.norm(ey)
            ez = np.cross(ex, ey)
        else:
            if abs(ex[2]) < 0.99:
                ez_temp = np.array([0, 0, 1])
            else:
                ez_temp = np.array([0, 1, 0])
            ey = np.cross(ez_temp, ex)
            ey = ey / np.linalg.norm(ey)
            ez = np.cross(ex, ey)
        T = np.zeros((12, 12))
        R = np.column_stack((ex, ey, ez))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        k_local = local_elastic_stiffness_matrix_3D_beam(elem['E'], elem['nu'], elem['A'], L, elem['Iy'], elem['Iz'], elem['J'])
        k_global = T.T @ k_local @ T
        dofs = np.concatenate((np.arange(6 * node_i, 6 * node_i + 6), np.arange(6 * node_j, 6 * node_j + 6)))
        for (i, gi) in enumerate(dofs):
            for (j, gj) in enumerate(dofs):
                K[gi, gj] += k_global[i, j]
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(n_dof)
    u[free_dofs] = u_f
    for elem in elements:
        (node_i, node_j) = (elem['node_i'], elem['node_j'])
        xi = node_coords[node_i]
        xj = node_coords[node_j]
        dx = xj - xi
        L = np.linalg.norm(dx)
        ex = dx / L
        if elem['local_z'] is not None:
            ez_temp = np.array(elem['local_z'])
            ey = np.cross(ez_temp, ex)
            ey = ey / np.linalg.norm(ey)
            ez = np.cross(ex, ey)
        else:
            if abs(ex[2]) < 0.99:
                ez_temp = np.array([0, 0, 1])
            else:
                ez_temp = np.array([0, 1, 0])
            ey = np.cross(ez_temp, ex)
            ey = ey / np.linalg.norm(ey)
            ez = np.cross(ex, ey)
        T = np.zeros((12, 12))
        R = np.column_stack((ex, ey, ez))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        dofs = np.concatenate((np.arange(6 * node_i, 6 * node_i + 6), np.arange(6 * node_j, 6 * node_j + 6)))
        u_elem = u[dofs]
        u_local = T @ u_elem
        f_local = k_local @ u_local
        Fx2 = f_local[6]
        Mx2 = f_local[9]
        (My1, Mz1) = (f_local[4], f_local[5])
        (My2, Mz2) = (f_local[10], f_local[11])
        kg_local = local_geometric_stiffness_matrix_3D_beam(L, elem['A'], elem['I_rho'], Fx2, Mx2, My1, Mz1, My2, Mz2)
        kg_global = T.T @ kg_local @ T
        for (i, gi) in enumerate(dofs):
            for (j, gj) in enumerate(dofs):
                K_g[gi, gj] += kg_global[i, j]
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    Kg_ff = K_g[np.ix_(free_dofs, free_dofs)]
    (eigenvals, eigenvecs) = scipy.linalg.eigh(K_ff, -Kg_ff)
    pos_eigenvals = eigenvals[eigenvals > 0]
    if len(pos_eigenvals) == 0:
        raise ValueError('No positive eigenvalues found')
    critical_load_factor = pos_eigenvals[0]
    mode_index = np.where(eigenvals == critical_load_factor)[0][0]
    mode_shape = np.zeros(n_dof)
    mode_shape[free_dofs] = eigenvecs[:, mode_index]
    return (critical_load_factor, mode_shape)