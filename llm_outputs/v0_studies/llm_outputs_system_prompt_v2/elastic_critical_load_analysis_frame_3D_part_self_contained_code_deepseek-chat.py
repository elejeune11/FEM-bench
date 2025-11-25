def elastic_critical_load_analysis_frame_3D_part_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    n_dof_total = 6 * n_nodes

    def get_transformation_matrix(node_i, node_j, local_z):
        vec_x = node_coords[node_j] - node_coords[node_i]
        L = np.linalg.norm(vec_x)
        e_x = vec_x / L
        if local_z is not None:
            e_z = np.array(local_z, dtype=float)
            e_z = e_z / np.linalg.norm(e_z)
            e_y = np.cross(e_z, e_x)
            e_y = e_y / np.linalg.norm(e_y)
            e_z = np.cross(e_x, e_y)
        else:
            if abs(e_x[2]) > 0.9:
                e_y = np.array([0.0, 1.0, 0.0])
            else:
                e_y = np.array([0.0, 0.0, 1.0])
            e_z = np.cross(e_x, e_y)
            e_z = e_z / np.linalg.norm(e_z)
            e_y = np.cross(e_z, e_x)
        R = np.zeros((3, 3))
        R[:, 0] = e_x
        R[:, 1] = e_y
        R[:, 2] = e_z
        T = np.zeros((12, 12))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        return (T, L)
    constrained_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        if isinstance(bc_spec[0], bool):
            for dof_local in range(6):
                if bc_spec[dof_local]:
                    constrained_dofs.add(6 * node_idx + dof_local)
        else:
            for dof_local in bc_spec:
                constrained_dofs.add(6 * node_idx + dof_local)
    free_dofs = sorted(set(range(n_dof_total)) - constrained_dofs)
    n_free = len(free_dofs)
    K_global = np.zeros((n_dof_total, n_dof_total))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        Iy = elem['Iy']
        Iz = elem['Iz']
        J = elem['J']
        local_z = elem.get('local_z', None)
        (T, L) = get_transformation_matrix(node_i, node_j, local_z)
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        k_global_elem = T.T @ k_local @ T
        dof_indices = []
        for node in [node_i, node_j]:
            for dof in range(6):
                dof_indices.append(6 * node + dof)
        for (i, dof_i) in enumerate(dof_indices):
            for (j, dof_j) in enumerate(dof_indices):
                K_global[dof_i, dof_j] += k_global_elem[i, j]
    P_global = np.zeros(n_dof_total)
    for (node_idx, load_vec) in nodal_loads.items():
        start_dof = 6 * node_idx
        P_global[start_dof:start_dof + 6] = load_vec
    K_free = K_global[np.ix_(free_dofs, free_dofs)]
    P_free = P_global[free_dofs]
    u_free = scipy.linalg.solve(K_free, P_free, assume_a='sym')
    u_global = np.zeros(n_dof_total)
    u_global[free_dofs] = u_free
    K_g_global = np.zeros((n_dof_total, n_dof_total))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        A = elem['A']
        I_rho = elem['I_rho']
        local_z = elem.get('local_z', None)
        (T, L) = get_transformation_matrix(node_i, node_j, local_z)
        dof_indices = [6 * node_i + dof for dof in range(6)] + [6 * node_j + dof for dof in range(6)]
        u_local = T @ u_global[dof_indices]
        Fx2 = -u_local[6] * elem['E'] * elem['A'] / L
        Mx2 = 0.0
        My1 = 0.0
        Mz1 = 0.0
        My2 = 0.0
        Mz2 = 0.0
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global_elem = T.T @ k_g_local @ T
        for (i, dof_i) in enumerate(dof_indices):
            for (j, dof_j) in enumerate(dof_indices):
                K_g_global[dof_i, dof_j] += k_g_global_elem[i, j]
    K_free = K_global[np.ix_(free_dofs, free_dofs)]
    K_g_free = K_g_global[np.ix_(free_dofs, free_dofs)]
    (eigvals, eigvecs_free) = scipy.linalg.eigh(K_free, -K_g_free, subset_by_index=[0, min(5, n_free - 1)])
    positive_eigvals = eigvals[eigvals > 0]
    if len(positive_eigvals) == 0:
        raise ValueError('No positive eigenvalues found')
    min_idx = np.argmin(positive_eigvals)
    critical_load_factor = positive_eigvals[min_idx]
    mode_free = eigvecs_free[:, np.where(eigvals == positive_eigvals[min_idx])[0][0]]
    deformed_shape_vector = np.zeros(n_dof_total)
    deformed_shape_vector[free_dofs] = mode_free
    return (critical_load_factor, deformed_shape_vector)