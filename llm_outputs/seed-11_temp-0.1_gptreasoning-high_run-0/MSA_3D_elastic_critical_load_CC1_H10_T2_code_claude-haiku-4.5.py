def MSA_3D_elastic_critical_load_CC1_H10_T2(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    K_global = np.zeros((n_dof, n_dof))
    P_global = np.zeros(n_dof)

    def compute_transformation_matrix(node_i_coords, node_j_coords, local_z=None):
        x_local = node_j_coords - node_i_coords
        L = np.linalg.norm(x_local)
        x_local = x_local / L
        if local_z is None:
            if abs(x_local[2]) < 0.9:
                local_z = np.array([0.0, 0.0, 1.0])
            else:
                local_z = np.array([0.0, 1.0, 0.0])
        else:
            local_z = np.array(local_z) / np.linalg.norm(local_z)
        local_z = local_z - np.dot(local_z, x_local) * x_local
        local_z = local_z / np.linalg.norm(local_z)
        y_local = np.cross(local_z, x_local)
        y_local = y_local / np.linalg.norm(y_local)
        R = np.array([x_local, y_local, local_z]).T
        T = np.zeros((12, 12))
        for i in range(2):
            T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
            T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
        return (T, L, R)

    def local_elastic_stiffness_matrix_3D_beam(L, E, A, I_y, I_z, J, nu):
        G = E / (2.0 * (1.0 + nu))
        k_e = np.zeros((12, 12))
        k_e[0, 0] = E * A / L
        k_e[0, 6] = -E * A / L
        k_e[6, 0] = -E * A / L
        k_e[6, 6] = E * A / L
        k_e[3, 3] = G * J / L
        k_e[3, 9] = -G * J / L
        k_e[9, 3] = -G * J / L
        k_e[9, 9] = G * J / L
        k_e[2, 2] = 12.0 * E * I_z / L ** 3
        k_e[2, 5] = 6.0 * E * I_z / L ** 2
        k_e[2, 8] = -12.0 * E * I_z / L ** 3
        k_e[2, 11] = 6.0 * E * I_z / L ** 2
        k_e[5, 2] = 6.0 * E * I_z / L ** 2
        k_e[5, 5] = 4.0 * E * I_z / L
        k_e[5, 8] = -6.0 * E * I_z / L ** 2
        k_e[5, 11] = 2.0 * E * I_z / L
        k_e[8, 2] = -12.0 * E * I_z / L ** 3
        k_e[8, 5] = -6.0 * E * I_z / L ** 2
        k_e[8, 8] = 12.0 * E * I_z / L ** 3
        k_e[8, 11] = -6.0 * E * I_z / L ** 2
        k_e[11, 2] = 6.0 * E * I_z / L ** 2
        k_e[11, 5] = 2.0 * E * I_z / L
        k_e[11, 8] = -6.0 * E * I_z / L ** 2
        k_e[11, 11] = 4.0 * E * I_z / L
        k_e[1, 1] = 12.0 * E * I_y / L ** 3
        k_e[1, 4] = -6.0 * E * I_y / L ** 2
        k_e[1, 7] = -12.0 * E * I_y / L ** 3
        k_e[1, 10] = -6.0 * E * I_y / L ** 2
        k_e[4, 1] = -6.0 * E * I_y / L ** 2
        k_e[4, 4] = 4.0 * E * I_y / L
        k_e[4, 7] = 6.0 * E * I_y / L ** 2
        k_e[4, 10] = 2.0 * E * I_y / L
        k_e[7, 1] = -12.0 * E * I_y / L ** 3
        k_e[7, 4] = 6.0 * E * I_y / L ** 2
        k_e[7, 7] = 12.0 * E * I_y / L ** 3
        k_e[7, 10] = 6.0 * E * I_y / L ** 2
        k_e[10, 1] = -6.0 * E * I_y / L ** 2
        k_e[10, 4] = 2.0 * E * I_y / L
        k_e[10, 7] = 6.0 * E * I_y / L ** 2
        k_e[10, 10] = 4.0 * E * I_y / L
        return k_e
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        local_z = elem.get('local_z', None)
        node_i_coords = node_coords[node_i]
        node_j_coords = node_coords[node_j]
        (T, L, R) = compute_transformation_matrix(node_i_coords, node_j_coords, local_z)
        k_e_local = local_elastic_stiffness_matrix_3D_beam(L, E, A, I_y, I_z, J, nu)
        k_e_global = T.T @ k_e_local @ T
        dof_i = np.arange(6 * node_i, 6 * node_i + 6)
        dof_j = np.arange(6 * node_j, 6 * node_j + 6)
        dof_indices = np.concatenate([dof_i, dof_j])
        for (ii, dof_ii) in enumerate(dof_indices):
            for (jj, dof_jj) in enumerate(dof_indices):
                K_global[dof_ii, dof_jj] += k_e_global[ii, jj]
    for (node_idx, loads) in nodal_loads.items():
        dof_indices = np.arange(6 * node_idx, 6 * node_idx + 6)
        P_global[dof_indices] = loads
    constrained_dofs = []
    for (node_idx, bc) in boundary_conditions.items():
        for (dof_local, is_fixed) in enumerate(bc):
            if is_fixed:
                constrained_dofs.append(6 * node_idx + dof_local)
    constrained_dofs = np.array(constrained_dofs, dtype=int)
    free_dofs = np.setdiff1d(np.arange(n_dof), constrained_dofs)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    P_f = P_global[free_dofs]
    u_f = np.linalg.solve(K_ff, P_f)
    u_global = np.zeros(n_dof)
    u_global[free_dofs] = u_f
    K_g_global = np.zeros((n_dof, n_dof))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        local_z = elem.get('local_z', None)
        node_i_coords = node_coords[node_i]
        node_j_coords = node_coords[node_j]
        (T, L, R) = compute_transformation_matrix(node_i_coords, node_j_coords, local_z)
        dof_i = np.arange(6 * node_i, 6 * node_i + 6)
        dof_j = np.arange(6 * node_j, 6 * node_j + 6)
        u_elem_global = np.concatenate([u_global[dof_i], u_global[dof_j]])
        u_elem_local = T @ u_elem_global
        k_e_local = local_elastic_stiffness_matrix_3D_beam(L, elem['E'], A, I_y, I_z, J, elem['nu'])
        f_elem_local = k_e_local @ u_elem_local
        Fx2 = f_elem_local[6]
        Fy2 = f_elem_local[7]
        Fz2 = f_elem_local[8]
        Mx2 = f_elem_local[9]
        My2 = f_elem_local[10]
        Mz2 = f_elem_local[11]
        Mx1 = f_elem_local[3]
        My1 = f_elem_local[4]
        Mz1 = f_elem_local[5]
        I_rho = I_y + I_z
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = T.T @ k_g_local @ T
        dof_indices = np.concatenate([dof_i, dof_j])
        for (ii, dof_ii) in enumerate(dof_indices):
            for (jj, dof_jj) in enumerate(dof_indices):
                K_g_global[dof_ii, dof_jj] += k_g_global[ii, jj]
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_g_ff = K_g_global[np.ix_(free_dofs, free_dofs)]
    (eigenvalues, eigenvectors) = scipy.linalg.eig(K_ff, -K_g_ff)
    real_eigenvalues = np.real(eigenvalues[np.abs(np.imag(eigenvalues)) < 1e-10])
    positive_eigenvalues = real_eigenvalues[real_eigenvalues > 1e-10]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalue found in buckling analysis')
    min_idx = np.argmin(positive_eigenvalues)
    elastic_critical_load_factor = positive_eigenvalues[min_idx]
    eigenvalue_idx = np.where(np.abs(eigenvalues - elastic_critical_load_factor) < 1e-08)[0][0]
    mode_vector_free = np.real(eigenvectors[:, eigenvalue_idx])
    deformed_shape_vector = np.zeros(n_dof)
    deformed_shape_vector[free_dofs] = mode_vector_free
    return (elastic_critical_load_factor, deformed_shape_vector)