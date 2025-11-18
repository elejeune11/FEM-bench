def elastic_critical_load_analysis_frame_3D_part_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K = np.zeros((n_dofs, n_dofs))
    P = np.zeros(n_dofs)
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        E = element['E']
        nu = element['nu']
        A = element['A']
        Iy = element['Iy']
        Iz = element['Iz']
        J = element['J']
        I_rho = element['I_rho']
        local_z = element['local_z']
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        if local_z is None:
            local_z = np.array([0, 0, 1])
        local_x = np.array([xj - xi, yj - yi, zj - zi]) / L
        local_y = np.cross(local_z, local_x)
        local_z = np.cross(local_x, local_y)
        T = np.zeros((12, 12))
        T[0:3, 0:3] = T[6:9, 6:9] = np.vstack([local_x, local_y, local_z]).T
        T[3:6, 3:6] = T[9:12, 9:12] = np.vstack([local_x, local_y, local_z]).T
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        k_local = T.T @ k_local @ T
        dof_map = np.array([6 * node_i, 6 * node_i + 1, 6 * node_i + 2, 6 * node_i + 3, 6 * node_i + 4, 6 * node_i + 5, 6 * node_j, 6 * node_j + 1, 6 * node_j + 2, 6 * node_j + 3, 6 * node_j + 4, 6 * node_j + 5])
        for a in range(12):
            for b in range(12):
                K[dof_map[a], dof_map[b]] += k_local[a, b]
    for (node, load) in nodal_loads.items():
        for i in range(6):
            P[6 * node + i] = load[i]
    free_dofs = np.ones(n_dofs, dtype=bool)
    for (node, bc) in boundary_conditions.items():
        if isinstance(bc, Sequence) and all((isinstance(x, bool) for x in bc)):
            for (i, fixed) in enumerate(bc):
                if fixed:
                    free_dofs[6 * node + i] = False
        elif isinstance(bc, Sequence) and all((isinstance(x, int) for x in bc)):
            for i in bc:
                free_dofs[6 * node + i] = False
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    u_f = np.linalg.solve(K_ff, P_f)
    K_g = np.zeros((n_dofs, n_dofs))
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        A = element['A']
        I_rho = element['I_rho']
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        u_i = np.zeros(6)
        u_j = np.zeros(6)
        for i in range(6):
            if free_dofs[6 * node_i + i]:
                u_i[i] = u_f[np.where(free_dofs)[0] == 6 * node_i + i]
            if free_dofs[6 * node_j + i]:
                u_j[i] = u_f[np.where(free_dofs)[0] == 6 * node_j + i]
        Fx2 = (u_j[0] - u_i[0]) * E * A / L
        Mx2 = (u_j[3] - u_i[3]) * E * J / (2.0 * (1.0 + nu) * L)
        My1 = (u_i[4] - u_j[4]) * E * Iy / L
        Mz1 = (u_i[5] - u_j[5]) * E * Iz / L
        My2 = (u_j[4] - u_i[4]) * E * Iy / L
        Mz2 = (u_j[5] - u_i[5]) * E * Iz / L
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_local = T.T @ k_g_local @ T
        for a in range(12):
            for b in range(12):
                K_g[dof_map[a], dof_map[b]] += k_g_local[a, b]
    K_g_ff = K_g[np.ix_(free_dofs, free_dofs)]
    (eigenvalues, eigenvectors) = eigh(K_ff, -K_g_ff)
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalue found.')
    elastic_critical_load_factor = positive_eigenvalues[0]
    mode_shape_f = eigenvectors[:, np.where(eigenvalues == elastic_critical_load_factor)[0][0]]
    deformed_shape_vector = np.zeros(n_dofs)
    deformed_shape_vector[free_dofs] = mode_shape_f
    return (elastic_critical_load_factor, deformed_shape_vector)