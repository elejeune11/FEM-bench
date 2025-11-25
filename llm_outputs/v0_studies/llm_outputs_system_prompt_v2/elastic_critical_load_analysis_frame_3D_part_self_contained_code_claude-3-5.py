def elastic_critical_load_analysis_frame_3D_part_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    K_g = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    for (node_idx, loads) in nodal_loads.items():
        start_idx = 6 * node_idx
        P[start_idx:start_idx + 6] = loads
    for elem in elements:
        (node_i, node_j) = (elem['node_i'], elem['node_j'])
        (x_i, y_i, z_i) = node_coords[node_i]
        (x_j, y_j, z_j) = node_coords[node_j]
        dx = x_j - x_i
        dy = y_j - y_i
        dz = z_j - z_i
        L = np.sqrt(dx * dx + dy * dy + dz * dz)
        ex = np.array([dx, dy, dz]) / L
        if elem['local_z'] is not None:
            ez_temp = np.array(elem['local_z'])
            ez = ez_temp - np.dot(ez_temp, ex) * ex
            ez = ez / np.linalg.norm(ez)
        else:
            if abs(ex[2]) < 0.9:
                ez_temp = np.array([0, 0, 1])
            else:
                ez_temp = np.array([0, 1, 0])
            ez = ez_temp - np.dot(ez_temp, ex) * ex
            ez = ez / np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        T = np.zeros((12, 12))
        R = np.vstack((ex, ey, ez)).T
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        k_local = local_elastic_stiffness_matrix_3D_beam(elem['E'], elem['nu'], elem['A'], L, elem['Iy'], elem['Iz'], elem['J'])
        k_global = T.T @ k_local @ T
        dofs = np.concatenate((np.arange(6 * node_i, 6 * node_i + 6), np.arange(6 * node_j, 6 * node_j + 6)))
        for (i, gi) in enumerate(dofs):
            for (j, gj) in enumerate(dofs):
                K[gi, gj] += k_global[i, j]
    free_dofs = np.ones(n_dof, dtype=bool)
    for (node, bc) in boundary_conditions.items():
        if isinstance(bc[0], bool):
            start_idx = 6 * node
            free_dofs[start_idx:start_idx + 6] = ~np.array(bc)
        else:
            start_idx = 6 * node
            free_dofs[start_idx + np.array(bc)] = False
    K_free = K[free_dofs][:, free_dofs]
    P_free = P[free_dofs]
    u_free = np.linalg.solve(K_free, P_free)
    u = np.zeros(n_dof)
    u[free_dofs] = u_free
    for elem in elements:
        (node_i, node_j) = (elem['node_i'], elem['node_j'])
        (x_i, y_i, z_i) = node_coords[node_i]
        (x_j, y_j, z_j) = node_coords[node_j]
        dx = x_j - x_i
        dy = y_j - y_i
        dz = z_j - z_i
        L = np.sqrt(dx * dx + dy * dy + dz * dz)
        ex = np.array([dx, dy, dz]) / L
        if elem['local_z'] is not None:
            ez_temp = np.array(elem['local_z'])
            ez = ez_temp - np.dot(ez_temp, ex) * ex
            ez = ez / np.linalg.norm(ez)
        else:
            if abs(ex[2]) < 0.9:
                ez_temp = np.array([0, 0, 1])
            else:
                ez_temp = np.array([0, 1, 0])
            ez = ez_temp - np.dot(ez_temp, ex) * ex
            ez = ez / np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        T = np.zeros((12, 12))
        R = np.vstack((ex, ey, ez)).T
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        u_elem = np.concatenate((u[6 * node_i:6 * node_i + 6], u[6 * node_j:6 * node_j + 6]))
        u_local = T @ u_elem
        f_local = k_local @ u_local
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, elem['A'], elem['I_rho'], f_local[6], f_local[9], f_local[4], f_local[5], f_local[10], f_local[11])
        k_g_global = T.T @ k_g_local @ T
        dofs = np.concatenate((np.arange(6 * node_i, 6 * node_i + 6), np.arange(6 * node_j, 6 * node_j + 6)))
        for (i, gi) in enumerate(dofs):
            for (j, gj) in enumerate(dofs):
                K_g[gi, gj] += k_g_global[i, j]
    K_g_free = K_g[free_dofs][:, free_dofs]
    (eigvals, eigvecs) = scipy.linalg.eigh(K_free, -K_g_free)
    pos_eigvals = eigvals[eigvals > 0]
    if len(pos_eigvals) == 0:
        raise ValueError('No positive eigenvalues found')
    critical_load_factor = pos_eigvals[0]
    mode_shape = np.zeros(n_dof)
    mode_shape[free_dofs] = eigvecs[:, np.where(eigvals == critical_load_factor)[0][0]]
    return (critical_load_factor, mode_shape)