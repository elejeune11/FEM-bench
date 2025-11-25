def elastic_critical_load_analysis_frame_3D_part_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    for (node_idx, loads) in nodal_loads.items():
        start_dof = 6 * node_idx
        P[start_dof:start_dof + 6] = loads
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        dx = xj - xi
        dy = yj - yi
        dz = zj - zi
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        k_local = local_elastic_stiffness_matrix_3D_beam(elem['E'], elem['nu'], elem['A'], L, elem['I_y'], elem['I_z'], elem['J'])
        x_local = np.array([dx, dy, dz]) / L
        if elem.get('local_z') is not None:
            z_ref = np.array(elem['local_z'])
            y_local = np.cross(z_ref, x_local)
            y_local = y_local / np.linalg.norm(y_local)
            z_local = np.cross(x_local, y_local)
        else:
            if abs(x_local[2]) < 0.9:
                z_ref = np.array([0, 0, 1])
            else:
                z_ref = np.array([1, 0, 0])
            y_local = np.cross(z_ref, x_local)
            y_local = y_local / np.linalg.norm(y_local)
            z_local = np.cross(x_local, y_local)
        R = np.array([x_local, y_local, z_local]).T
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        k_global = T.T @ k_local @ T
        dof_i = [6 * node_i + k for k in range(6)]
        dof_j = [6 * node_j + k for k in range(6)]
        dofs = dof_i + dof_j
        for (i, dof_i) in enumerate(dofs):
            for (j, dof_j) in enumerate(dofs):
                K[dof_i, dof_j] += k_global[i, j]
    constrained_dofs = set()
    for (node_idx, bc) in boundary_conditions.items():
        base_dof = 6 * node_idx
        if all((isinstance(b, bool) for b in bc)):
            for (i, is_fixed) in enumerate(bc):
                if is_fixed:
                    constrained_dofs.add(base_dof + i)
        else:
            for dof_idx in bc:
                constrained_dofs.add(base_dof + dof_idx)
    free_dofs = [i for i in range(n_dof) if i not in constrained_dofs]
    n_free = len(free_dofs)
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(n_dof)
    u[free_dofs] = u_f
    K_g = np.zeros((n_dof, n_dof))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        dx = xj - xi
        dy = yj - yi
        dz = zj - zi
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        x_local = np.array([dx, dy, dz]) / L
        if elem.get('local_z') is not None:
            z_ref = np.array(elem['local_z'])
            y_local = np.cross(z_ref, x_local)
            y_local = y_local / np.linalg.norm(y_local)
            z_local = np.cross(x_local, y_local)
        else:
            if abs(x_local[2]) < 0.9:
                z_ref = np.array([0, 0, 1])
            else:
                z_ref = np.array([1, 0, 0])
            y_local = np.cross(z_ref, x_local)
            y_local = y_local / np.linalg.norm(y_local)
            z_local = np.cross(x_local, y_local)
        R = np.array([x_local, y_local, z_local]).T
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        dof_i = [6 * node_i + k for k in range(6)]
        dof_j = [6 * node_j + k for k in range(6)]
        dofs = dof_i + dof_j
        u_elem_global = u[dofs]
        u_elem_local = T @ u_elem_global
        k_local = local_elastic_stiffness_matrix_3D_beam(elem['E'], elem['nu'], elem['A'], L, elem['I_y'], elem['I_z'], elem['J'])
        f_local = k_local @ u_elem_local
        Fx2 = -f_local[6]
        Mx2 = -f_local[9]
        My1 = f_local[4]
        Mz1 = f_local[5]
        My2 = -f_local[10]
        Mz2 = -f_local[11]
        kg_local = local_geometric_stiffness_matrix_3D_beam(L, elem['A'], elem['I_rho'], Fx2, Mx2, My1, Mz1, My2, Mz2)
        kg_global = T.T @ kg_local @ T
        for (i, dof_i) in enumerate(dofs):
            for (j, dof_j) in enumerate(dofs):
                K_g[dof_i, dof_j] += kg_global[i, j]
    K_g_ff = K_g[np.ix_(free_dofs, free_dofs)]
    (eigenvalues, eigenvectors) = scipy.linalg.eig(K_ff, -K_g_ff)
    real_eigenvalues = eigenvalues.real
    positive_mask = real_eigenvalues > 1e-10
    positive_eigenvalues = real_eigenvalues[positive_mask]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalues found')
    min_idx_in_positive = np.argmin(positive_eigenvalues)
    min_eigenvalue = positive_eigenvalues[min_idx_in_positive]
    original_idx = np.where(positive_mask)[0][min_idx_in_positive]
    mode_free = eigenvectors[:, original_idx].real
    mode_shape = np.zeros(n_dof)
    mode_shape[free_dofs] = mode_free
    return (min_eigenvalue, mode_shape)