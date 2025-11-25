def elastic_critical_load_analysis_frame_3D_part_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K_global = np.zeros((n_dofs, n_dofs))
    P_global = np.zeros(n_dofs)
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
        direction_cosines = np.array([(xj - xi) / L, (yj - yi) / L, (zj - zi) / L])
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        if local_z is None:
            local_z = np.array([0, 0, 1])
        local_y = np.cross(local_z, direction_cosines)
        local_y /= np.linalg.norm(local_y)
        local_z = np.cross(direction_cosines, local_y)
        T = np.zeros((12, 12))
        T[0:3, 0:3] = T[6:9, 6:9] = np.vstack((direction_cosines, local_y, local_z)).T
        T[3:6, 3:6] = T[9:12, 9:12] = T[0:3, 0:3]
        k_global = T.T @ k_local @ T
        dof_indices = np.array([6 * node_i, 6 * node_i + 1, 6 * node_i + 2, 6 * node_i + 3, 6 * node_i + 4, 6 * node_i + 5, 6 * node_j, 6 * node_j + 1, 6 * node_j + 2, 6 * node_j + 3, 6 * node_j + 4, 6 * node_j + 5])
        for i in range(12):
            for j in range(12):
                K_global[dof_indices[i], dof_indices[j]] += k_global[i, j]
    for (node, loads) in nodal_loads.items():
        for i in range(6):
            P_global[6 * node + i] += loads[i]
    constrained_dofs = []
    for (node, bc) in boundary_conditions.items():
        if isinstance(bc[0], bool):
            constrained_dofs.extend([6 * node + i for (i, fixed) in enumerate(bc) if fixed])
        else:
            constrained_dofs.extend([6 * node + i for i in bc])
    free_dofs = np.setdiff1d(np.arange(n_dofs), constrained_dofs)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    P_f = P_global[free_dofs]
    u_f = np.linalg.solve(K_ff, P_f)
    K_g_global = np.zeros((n_dofs, n_dofs))
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        A = element['A']
        I_rho = element['I_rho']
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        direction_cosines = np.array([(xj - xi) / L, (yj - yi) / L, (zj - zi) / L])
        u_local = np.zeros(12)
        dof_indices = np.array([6 * node_i, 6 * node_i + 1, 6 * node_i + 2, 6 * node_i + 3, 6 * node_i + 4, 6 * node_i + 5, 6 * node_j, 6 * node_j + 1, 6 * node_j + 2, 6 * node_j + 3, 6 * node_j + 4, 6 * node_j + 5])
        for (i, dof) in enumerate(dof_indices):
            if dof in free_dofs:
                u_local[i] = u_f[np.where(free_dofs == dof)[0][0]]
        Fx2 = u_local[6] * E * A / L
        Mx2 = u_local[9] * E * J / (2.0 * (1.0 + nu) * L)
        My1 = u_local[4] * E * Iy / L
        Mz1 = u_local[5] * E * Iz / L
        My2 = u_local[10] * E * Iy / L
        Mz2 = u_local[11] * E * Iz / L
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = T.T @ k_g_local @ T
        for i in range(12):
            for j in range(12):
                K_g_global[dof_indices[i], dof_indices[j]] += k_g_global[i, j]
    K_g_ff = K_g_global[np.ix_(free_dofs, free_dofs)]
    (eigenvalues, eigenvectors) = eigs(K_ff, k=1, M=-K_g_ff, which='SM', sigma=0)
    lambda_crit = np.real(eigenvalues[0])
    phi_f = np.real(eigenvectors[:, 0])
    deformed_shape_vector = np.zeros(n_dofs)
    deformed_shape_vector[free_dofs] = phi_f
    return (lambda_crit, deformed_shape_vector)