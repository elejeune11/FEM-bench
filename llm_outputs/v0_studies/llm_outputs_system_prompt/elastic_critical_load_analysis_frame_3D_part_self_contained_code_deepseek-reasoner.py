def elastic_critical_load_analysis_frame_3D_part_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K = np.zeros((n_dofs, n_dofs))
    P = np.zeros(n_dofs)
    for element in elements:
        (i, j) = (element['node_i'], element['node_j'])
        (coords_i, coords_j) = (node_coords[i], node_coords[j])
        L = np.linalg.norm(coords_j - coords_i)
        k_local = local_elastic_stiffness_matrix_3D_beam(element['E'], element['nu'], element['A'], L, element['Iy'], element['Iz'], element['J'])
        if element.get('local_z') is not None:
            local_z = np.array(element['local_z'])
            local_x = (coords_j - coords_i) / L
            local_y = np.cross(local_z, local_x)
            local_y /= np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
        else:
            local_x = (coords_j - coords_i) / L
            if abs(local_x[2]) > 0.9:
                local_y = np.array([0.0, 1.0, 0.0])
            else:
                local_y = np.array([0.0, 0.0, 1.0])
            local_y = local_y - np.dot(local_y, local_x) * local_x
            local_y /= np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
        R = np.column_stack([local_x, local_y, local_z])
        T = np.zeros((12, 12))
        for block in range(4):
            T[3 * block:3 * block + 3, 3 * block:3 * block + 3] = R
        k_global = T.T @ k_local @ T
        dofs_i = slice(6 * i, 6 * i + 6)
        dofs_j = slice(6 * j, 6 * j + 6)
        dofs = np.r_[dofs_i, dofs_j]
        K[np.ix_(dofs, dofs)] += k_global
    for (node, load) in nodal_loads.items():
        P[6 * node:6 * node + 6] = load
    constrained_dofs = []
    for (node, bc) in boundary_conditions.items():
        if all((isinstance(x, bool) for x in bc)):
            for (dof_idx, is_constrained) in enumerate(bc):
                if is_constrained:
                    constrained_dofs.append(6 * node + dof_idx)
        else:
            for dof_idx in bc:
                constrained_dofs.append(6 * node + dof_idx)
    free_dofs = [i for i in range(n_dofs) if i not in constrained_dofs]
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    u_f = scipy.linalg.solve(K_ff, P_f, assume_a='sym')
    u = np.zeros(n_dofs)
    u[free_dofs] = u_f
    K_g = np.zeros((n_dofs, n_dofs))
    for element in elements:
        (i, j) = (element['node_i'], element['node_j'])
        (coords_i, coords_j) = (node_coords[i], node_coords[j])
        L = np.linalg.norm(coords_j - coords_i)
        if element.get('local_z') is not None:
            local_z = np.array(element['local_z'])
            local_x = (coords_j - coords_i) / L
            local_y = np.cross(local_z, local_x)
            local_y /= np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
        else:
            local_x = (coords_j - coords_i) / L
            if abs(local_x[2]) > 0.9:
                local_y = np.array([0.0, 1.0, 0.0])
            else:
                local_y = np.array([0.0, 0.0, 1.0])
            local_y = local_y - np.dot(local_y, local_x) * local_x
            local_y /= np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
        R = np.column_stack([local_x, local_y, local_z])
        T = np.zeros((12, 12))
        for block in range(4):
            T[3 * block:3 * block + 3, 3 * block:3 * block + 3] = R
        u_local = T.T @ np.r_[u[6 * i:6 * i + 6], u[6 * j:6 * j + 6]]
        k_local_elastic = local_elastic_stiffness_matrix_3D_beam(element['E'], element['nu'], element['A'], L, element['Iy'], element['Iz'], element['J'])
        f_local = k_local_elastic @ u_local
        Fx2 = f_local[6]
        Mx2 = f_local[9]
        My1 = f_local[4]
        Mz1 = f_local[5]
        My2 = f_local[10]
        Mz2 = f_local[11]
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, element['A'], element['I_rho'], Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = T.T @ k_g_local @ T
        dofs_i = slice(6 * i, 6 * i + 6)
        dofs_j = slice(6 * j, 6 * j + 6)
        dofs = np.r_[dofs_i, dofs_j]
        K_g[np.ix_(dofs, dofs)] += k_g_global
    K_g_ff = K_g[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_ff, -K_g_ff)
    except (scipy.linalg.LinAlgError, ValueError):
        (eigenvalues, eigenvectors) = scipy.linalg.eig(K_ff, -K_g_ff)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalues found')
    min_idx = np.argmin(positive_eigenvalues)
    elastic_critical_load_factor = positive_eigenvalues[min_idx]
    original_idx = np.where(eigenvalues == positive_eigenvalues[min_idx])[0][0]
    mode_f = eigenvectors[:, original_idx]
    deformed_shape_vector = np.zeros(n_dofs)
    deformed_shape_vector[free_dofs] = mode_f
    return (elastic_critical_load_factor, deformed_shape_vector)