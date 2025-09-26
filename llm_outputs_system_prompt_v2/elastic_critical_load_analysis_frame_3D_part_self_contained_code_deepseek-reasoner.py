def elastic_critical_load_analysis_frame_3D_part_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    constrained_dofs = set()
    for (node_idx, bc) in boundary_conditions.items():
        if all((isinstance(x, bool) for x in bc)):
            for (dof_idx, is_constrained) in enumerate(bc):
                if is_constrained:
                    constrained_dofs.add(6 * node_idx + dof_idx)
        else:
            for dof_idx in bc:
                constrained_dofs.add(6 * node_idx + dof_idx)
    free_dofs = sorted(set(range(n_dofs)) - constrained_dofs)
    n_free = len(free_dofs)
    K_global = np.zeros((n_dofs, n_dofs))
    P_global = np.zeros(n_dofs)
    for (node_idx, load) in nodal_loads.items():
        start_dof = 6 * node_idx
        P_global[start_dof:start_dof + 6] = load
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        L = np.linalg.norm(node_coords[j] - node_coords[i])
        k_local = local_elastic_stiffness_matrix_3D_beam(elem['E'], elem['nu'], elem['A'], L, elem['Iy'], elem['Iz'], elem['J'])
        vec_x = node_coords[j] - node_coords[i]
        e_x = vec_x / L
        if elem.get('local_z') is not None:
            e_z = np.array(elem['local_z'])
            e_z = e_z - np.dot(e_z, e_x) * e_x
            e_z = e_z / np.linalg.norm(e_z)
        else:
            if abs(e_x[0]) > 0.9:
                temp = np.array([0, 1, 0])
            else:
                temp = np.array([1, 0, 0])
            e_z = temp - np.dot(temp, e_x) * e_x
            e_z = e_z / np.linalg.norm(e_z)
        e_y = np.cross(e_z, e_x)
        e_y = e_y / np.linalg.norm(e_y)
        R = np.column_stack((e_x, e_y, e_z))
        T = np.zeros((12, 12))
        for block in range(4):
            (start, end) = (3 * block, 3 * block + 3)
            T[start:end, start:end] = R
        k_global_elem = T @ k_local @ T.T
        dof_indices = []
        for node in [i, j]:
            dof_indices.extend(range(6 * node, 6 * node + 6))
        for (idx1, global_idx1) in enumerate(dof_indices):
            for (idx2, global_idx2) in enumerate(dof_indices):
                K_global[global_idx1, global_idx2] += k_global_elem[idx1, idx2]
    free_indices = [free_dofs.index(dof) for dof in free_dofs]
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    P_f = P_global[free_dofs]
    u_f = scipy.linalg.solve(K_ff, P_f, assume_a='sym')
    u_global = np.zeros(n_dofs)
    u_global[free_dofs] = u_f
    K_g_global = np.zeros((n_dofs, n_dofs))
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        L = np.linalg.norm(node_coords[j] - node_coords[i])
        vec_x = node_coords[j] - node_coords[i]
        e_x = vec_x / L
        if elem.get('local_z') is not None:
            e_z = np.array(elem['local_z'])
            e_z = e_z - np.dot(e_z, e_x) * e_x
            e_z = e_z / np.linalg.norm(e_z)
        else:
            if abs(e_x[0]) > 0.9:
                temp = np.array([0, 1, 0])
            else:
                temp = np.array([1, 0, 0])
            e_z = temp - np.dot(temp, e_x) * e_x
            e_z = e_z / np.linalg.norm(e_z)
        e_y = np.cross(e_z, e_x)
        e_y = e_y / np.linalg.norm(e_y)
        R = np.column_stack((e_x, e_y, e_z))
        T = np.zeros((12, 12))
        for block in range(4):
            (start, end) = (3 * block, 3 * block + 3)
            T[start:end, start:end] = R
        dof_indices = []
        for node in [i, j]:
            dof_indices.extend(range(6 * node, 6 * node + 6))
        u_local = T.T @ u_global[dof_indices]
        k_local_elastic = local_elastic_stiffness_matrix_3D_beam(elem['E'], elem['nu'], elem['A'], L, elem['Iy'], elem['Iz'], elem['J'])
        f_local = k_local_elastic @ u_local
        Fx2 = f_local[6]
        Mx2 = f_local[9]
        My1 = f_local[4]
        Mz1 = f_local[5]
        My2 = f_local[10]
        Mz2 = f_local[11]
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, elem['A'], elem['I_rho'], Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global_elem = T @ k_g_local @ T.T
        for (idx1, global_idx1) in enumerate(dof_indices):
            for (idx2, global_idx2) in enumerate(dof_indices):
                K_g_global[global_idx1, global_idx2] += k_g_global_elem[idx1, idx2]
    K_g_ff = K_g_global[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_ff, -K_g_ff)
    except:
        (eigenvalues, eigenvectors) = scipy.linalg.eig(K_ff, -K_g_ff)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalues found')
    min_positive_idx = np.argmin(positive_eigenvalues)
    critical_load_factor = positive_eigenvalues[min_positive_idx]
    original_idx = np.where(eigenvalues == positive_eigenvalues[min_positive_idx])[0][0]
    mode_shape_free = eigenvectors[:, original_idx]
    mode_shape = np.zeros(n_dofs)
    mode_shape[free_dofs] = mode_shape_free
    return (critical_load_factor, mode_shape)