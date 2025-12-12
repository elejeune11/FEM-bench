def MSA_3D_elastic_critical_load_CC1_H10_T2(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    for (node_idx, loads) in nodal_loads.items():
        if len(loads) != 6:
            raise ValueError('Each nodal load must have 6 components: [Fx, Fy, Fz, Mx, My, Mz]')
        start_dof = node_idx * 6
        P[start_dof:start_dof + 6] = loads
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        coord_i = node_coords[node_i]
        coord_j = node_coords[node_j]
        L = np.linalg.norm(coord_j - coord_i)
        if L < 1e-12:
            raise ValueError('Element length is zero')
        local_z = np.array(elem.get('local_z', [0, 0, 1]), dtype=float)
        if np.linalg.norm(local_z) < 1e-12:
            raise ValueError('local_z vector must have non-zero magnitude')
        local_z = local_z / np.linalg.norm(local_z)
        local_x = (coord_j - coord_i) / L
        if np.abs(np.dot(local_x, local_z)) > 1 - 1e-10:
            if np.abs(local_x[2]) > 0.9:
                local_z = np.array([0, 1, 0], dtype=float)
            else:
                local_z = np.array([0, 0, 1], dtype=float)
            local_z = local_z - np.dot(local_x, local_z) * local_x
            local_z = local_z / np.linalg.norm(local_z)
        local_y = np.cross(local_z, local_x)
        local_y = local_y / np.linalg.norm(local_y)
        T = np.zeros((6, 6))
        T[0:3, 0:3] = np.array([local_x, local_y, local_z]).T
        T[3:6, 3:6] = np.array([local_x, local_y, local_z]).T
        k_local = np.zeros((12, 12))
        k_axial = E * A / L
        k_local[0, 0] = k_axial
        k_local[0, 6] = -k_axial
        k_local[6, 0] = -k_axial
        k_local[6, 6] = k_axial
        k_torsion = E * J / L
        k_local[3, 3] = k_torsion
        k_local[3, 9] = -k_torsion
        k_local[9, 3] = -k_torsion
        k_local[9, 9] = k_torsion
        k_bend_y = 12 * E * I_z / L ** 3
        k_bend_y_theta = 6 * E * I_z / L ** 2
        k_bend_y_theta2 = 4 * E * I_z / L
        k_bend_y_theta1 = 2 * E * I_z / L
        k_local[1, 1] = k_bend_y
        k_local[1, 5] = k_bend_y_theta
        k_local[1, 7] = -k_bend_y
        k_local[1, 11] = k_bend_y_theta
        k_local[5, 1] = k_bend_y_theta
        k_local[5, 5] = k_bend_y_theta2
        k_local[5, 7] = -k_bend_y_theta
        k_local[5, 11] = k_bend_y_theta1
        k_local[7, 1] = -k_bend_y
        k_local[7, 5] = -k_bend_y_theta
        k_local[7, 7] = k_bend_y
        k_local[7, 11] = -k_bend_y_theta
        k_local[11, 1] = k_bend_y_theta
        k_local[11, 5] = k_bend_y_theta1
        k_local[11, 7] = -k_bend_y_theta
        k_local[11, 11] = k_bend_y_theta2
        k_bend_z = 12 * E * I_y / L ** 3
        k_bend_z_theta = 6 * E * I_y / L ** 2
        k_bend_z_theta2 = 4 * E * I_y / L
        k_bend_z_theta1 = 2 * E * I_y / L
        k_local[2, 2] = k_bend_z
        k_local[2, 4] = -k_bend_z_theta
        k_local[2, 8] = -k_bend_z
        k_local[2, 10] = -k_bend_z_theta
        k_local[4, 2] = -k_bend_z_theta
        k_local[4, 4] = k_bend_z_theta2
        k_local[4, 8] = k_bend_z_theta
        k_local[4, 10] = -k_bend_z_theta1
        k_local[8, 2] = -k_bend_z
        k_local[8, 4] = k_bend_z_theta
        k_local[8, 8] = k_bend_z
        k_local[8, 10] = k_bend_z_theta
        k_local[10, 2] = -k_bend_z_theta
        k_local[10, 4] = -k_bend_z_theta1
        k_local[10, 8] = k_bend_z_theta
        k_local[10, 10] = k_bend_z_theta2
        K_global = T.T @ k_local @ T
        dof_i = [node_i * 6 + i for i in range(6)]
        dof_j = [node_j * 6 + i for i in range(6)]
        all_dofs = dof_i + dof_j
        for (i, di) in enumerate(all_dofs):
            for (j, dj) in enumerate(all_dofs):
                K[di, dj] += K_global[i, j]
    fixed_dofs = []
    for (node_idx, bc) in boundary_conditions.items():
        if len(bc) != 6:
            raise ValueError('Each boundary condition must have 6 values (0/1 for each DOF)')
        for (i, fixed) in enumerate(bc):
            if fixed == 1:
                fixed_dofs.append(node_idx * 6 + i)
    all_dofs = set(range(n_dof))
    fixed_set = set(fixed_dofs)
    free_dofs = sorted(list(all_dofs - fixed_set))
    if len(free_dofs) == 0:
        raise ValueError('No free DOFs remain after applying boundary conditions')
    K_reduced = K[np.ix_(free_dofs, free_dofs)]
    P_free = P[free_dofs]
    try:
        u_free = np.linalg.solve(K_reduced, P_free)
    except np.linalg.LinAlgError:
        raise ValueError('Stiffness matrix is singular after applying boundary conditions')
    u_full = np.zeros(n_dof)
    u_full[free_dofs] = u_free
    K_g = np.zeros((n_dof, n_dof))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        coord_i = node_coords[node_i]
        coord_j = node_coords[node_j]
        L = np.linalg.norm(coord_j - coord_i)
        local_z = np.array(elem.get('local_z', [0, 0, 1]), dtype=float)
        if np.linalg.norm(local_z) < 1e-12:
            raise ValueError('local_z vector must have non-zero magnitude')
        local_z = local_z / np.linalg.norm(local_z)
        local_x = (coord_j - coord_i) / L
        if np.abs(np.dot(local_x, local_z)) > 1 - 1e-10:
            if np.abs(local_x[2]) > 0.9:
                local_z = np.array([0, 1, 0], dtype=float)
            else:
                local_z = np.array([0, 0, 1], dtype=float)
            local_z = local_z - np.dot(local_x, local_z) * local_x
            local_z = local_z / np.linalg.norm(local_z)
        local_y = np.cross(local_z, local_x)
        local_y = local_y / np.linalg.norm(local_y)
        T_global_to_local = np.array([local_x, local_y, local_z]).T
        T = np.zeros((6, 6))
        T[0:3, 0:3] = T_global_to_local
        T[3:6, 3:6] = T_global_to_local
        u_i_global = u_full[node_i * 6:(node_i + 1) * 6]
        u_j_global = u_full[node_j * 6:(node_j + 1) * 6]
        u_i_local = T @ u_i_global
        u_j_local = T @ u_j_global
        Fx2 = E * A / L * (u_j_local[0] - u_i_local[0])
        G = E / (2 * (1 + nu))
        Mx2 = G * J / L * (u_j_local[3] - u_i_local[3])
        v1 = u_i_local[1]
        v2 = u_j_local[1]
        θz1 = u_i_local[5]
        θz2 = u_j_local[5]
        My1 = 2 * E * I_z / L * θz1 + E * I_z / L * θz2 - 6 * E * I_z / L ** 2 * (v1 - v2)
        My2 = E * I_z / L * θz1 + 2 * E * I_z / L * θz2 - 6 * E * I_z / L ** 2 * (v2 - v1)
        w1 = u_i_local[2]
        w2 = u_j_local[2]
        θy1 = u_i_local[4]
        θy2 = u_j_local[4]
        Mz1 = -(2 * E * I_y / L) * θy1 - E * I_y / L * θy2 + 6 * E * I_y / L ** 2 * (w1 - w2)
        Mz2 = -(E * I_y / L) * θy1 - 2 * E * I_y / L * θy2 + 6 * E * I_y / L ** 2 * (w2 - w1)
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, J, Fx2, Mx2, My1, Mz1, My2, Mz2)
        K_g_global = T.T @ k_g_local @ T
        dof_i = [node_i * 6 + i for i in range(6)]
        dof_j = [node_j * 6 + i for i in range(6)]
        all_dofs = dof_i + dof_j
        for (i, di) in enumerate(all_dofs):
            for (j, dj) in enumerate(all_dofs):
                K_g[di, dj] += K_g_global[i, j]
    K_g_reduced = K_g[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvals, eigenvecs) = scipy.linalg.eig(K_reduced, -K_g_reduced)
    except:
        raise ValueError('Eigenvalue problem could not be solved')
    real_eigenvals = np.real(eigenvals)
    positive_eigenvals = real_eigenvals[real_eigenvals > 1e-10]
    if len(positive_eigenvals) == 0:
        raise ValueError('No positive eigenvalues found')
    lambda_critical = np.min(positive_eigenvals)
    idx = np.argmin(real_eigenvals[real_eigenvals > 1e-10])
    real_positive_indices = np.where(real_eigenvals > 1e-10)[0]
    if len(real_positive_indices) == 0:
        raise ValueError('No positive eigenvalues found')
    eigenvector_idx = real_positive_indices[idx]
    phi_free = np.real(eigenvecs[:, eigenvector_idx])
    deformed_shape_vector = np.zeros(n_dof)
    deformed_shape_vector[free_dofs] = phi_free
    return (lambda_critical, deformed_shape_vector)