def elastic_critical_load_analysis_frame_3D_part_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    n_dof_total = 6 * n_nodes
    K_global = np.zeros((n_dof_total, n_dof_total))
    P_global = np.zeros(n_dof_total)
    for element in elements:
        i = element['node_i']
        j = element['node_j']
        E = element['E']
        nu = element['nu']
        A = element['A']
        Iy = element['Iy']
        Iz = element['Iz']
        J = element['J']
        I_rho = element['I_rho']
        node_i_coords = node_coords[i]
        node_j_coords = node_coords[j]
        L = np.linalg.norm(node_j_coords - node_i_coords)
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        if 'local_z' in element and element['local_z'] is not None:
            local_z = np.array(element['local_z'])
            local_z = local_z / np.linalg.norm(local_z)
            local_x = (node_j_coords - node_i_coords) / L
            local_y = np.cross(local_z, local_x)
            local_y = local_y / np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
        else:
            local_x = (node_j_coords - node_i_coords) / L
            if abs(local_x[2]) > 0.9:
                local_y = np.array([0.0, 1.0, 0.0])
            else:
                local_y = np.array([0.0, 0.0, 1.0])
            local_y = local_y - np.dot(local_y, local_x) * local_x
            local_y = local_y / np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
        R_small = np.column_stack([local_x, local_y, local_z])
        R = scipy.linalg.block_diag(R_small, R_small, R_small, R_small)
        k_global = R.T @ k_local @ R
        dof_i = [6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5]
        dof_j = [6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5]
        dofs = dof_i + dof_j
        for (idx_row, dof_row) in enumerate(dofs):
            for (idx_col, dof_col) in enumerate(dofs):
                K_global[dof_row, dof_col] += k_global[idx_row, idx_col]
    for (node_idx, load_vec) in nodal_loads.items():
        start_dof = 6 * node_idx
        P_global[start_dof:start_dof + 6] = load_vec
    constrained_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        if all((isinstance(x, bool) for x in bc_spec)):
            for (dof_local, is_fixed) in enumerate(bc_spec):
                if is_fixed:
                    constrained_dofs.add(6 * node_idx + dof_local)
        else:
            for dof_local in bc_spec:
                constrained_dofs.add(6 * node_idx + dof_local)
    free_dofs = [dof for dof in range(n_dof_total) if dof not in constrained_dofs]
    K_free = K_global[np.ix_(free_dofs, free_dofs)]
    P_free = P_global[free_dofs]
    u_free = scipy.linalg.solve(K_free, P_free, assume_a='sym')
    u_global = np.zeros(n_dof_total)
    u_global[free_dofs] = u_free
    K_g_global = np.zeros((n_dof_total, n_dof_total))
    for element in elements:
        i = element['node_i']
        j = element['node_j']
        A = element['A']
        I_rho = element['I_rho']
        node_i_coords = node_coords[i]
        node_j_coords = node_coords[j]
        L = np.linalg.norm(node_j_coords - node_i_coords)
        dof_i = [6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5]
        dof_j = [6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5]
        u_element = np.concatenate([u_global[dof_i], u_global[dof_j]])
        if 'local_z' in element and element['local_z'] is not None:
            local_z = np.array(element['local_z'])
            local_z = local_z / np.linalg.norm(local_z)
            local_x = (node_j_coords - node_i_coords) / L
            local_y = np.cross(local_z, local_x)
            local_y = local_y / np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
        else:
            local_x = (node_j_coords - node_i_coords) / L
            if abs(local_x[2]) > 0.9:
                local_y = np.array([0.0, 1.0, 0.0])
            else:
                local_y = np.array([0.0, 0.0, 1.0])
            local_y = local_y - np.dot(local_y, local_x) * local_x
            local_y = local_y / np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
        R_small = np.column_stack([local_x, local_y, local_z])
        R = scipy.linalg.block_diag(R_small, R_small, R_small, R_small)
        u_local = R @ u_element
        Fx2 = -E * A * (u_local[6] - u_local[0]) / L
        Mx2 = E * J / (2.0 * (1.0 + nu) * L) * (u_local[9] - u_local[3])
        My1 = 2.0 * E * Iy / L * (2.0 * u_local[4] + u_local[10])
        Mz1 = 2.0 * E * Iz / L * (2.0 * u_local[5] + u_local[11])
        My2 = 2.0 * E * Iy / L * (u_local[4] + 2.0 * u_local[10])
        Mz2 = 2.0 * E * Iz / L * (u_local[5] + 2.0 * u_local[11])
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = R.T @ k_g_local @ R
        dofs = dof_i + dof_j
        for (idx_row, dof_row) in enumerate(dofs):
            for (idx_col, dof_col) in enumerate(dofs):
                K_g_global[dof_row, dof_col] += k_g_global[idx_row, idx_col]
    K_g_free = K_g_global[np.ix_(free_dofs, free_dofs)]
    (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_free, -K_g_free)
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalues found')
    min_positive_idx = np.argmin(positive_eigenvalues)
    critical_load_factor = positive_eigenvalues[min_positive_idx]
    original_eigenvalue_idx = np.where(eigenvalues == critical_load_factor)[0][0]
    mode_free = eigenvectors[:, original_eigenvalue_idx]
    mode_global = np.zeros(n_dof_total)
    mode_global[free_dofs] = mode_free
    return (critical_load_factor, mode_global)