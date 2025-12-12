def MSA_3D_elastic_critical_load_CC1_H10_T2(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    K_global = np.zeros((n_dof, n_dof))
    P_global = np.zeros(n_dof)

    def compute_transformation_matrix(node_i_coords, node_j_coords, local_z=None):
        """Compute the 12x12 transformation matrix for a 3D beam element."""
        x_local = node_j_coords - node_i_coords
        L = np.linalg.norm(x_local)
        x_local = x_local / L
        if local_z is None:
            if abs(x_local[2]) < 0.9:
                local_z = np.array([0.0, 0.0, 1.0])
            else:
                local_z = np.array([0.0, 1.0, 0.0])
        else:
            local_z = np.array(local_z)
        local_z = local_z - np.dot(local_z, x_local) * x_local
        local_z = local_z / np.linalg.norm(local_z)
        y_local = np.cross(local_z, x_local)
        y_local = y_local / np.linalg.norm(y_local)
        R = np.array([x_local, y_local, local_z]).T
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
            T[3 * i + 6:3 * i + 9, 3 * i + 6:3 * i + 9] = R
        return (T, L, R)

    def local_elastic_stiffness_matrix(L, E, A, I_y, I_z, J):
        """Compute 12x12 local elastic stiffness matrix for 3D Euler-Bernoulli beam."""
        G = E / (2 * (1 + 0.3))
        k_e = np.zeros((12, 12))
        k_e[0, 0] = E * A / L
        k_e[0, 6] = -E * A / L
        k_e[6, 6] = E * A / L
        k_e[3, 3] = G * J / L
        k_e[3, 9] = -G * J / L
        k_e[9, 9] = G * J / L
        k_e[1, 1] = 12 * E * I_z / L ** 3
        k_e[1, 5] = 6 * E * I_z / L ** 2
        k_e[1, 7] = -12 * E * I_z / L ** 3
        k_e[1, 11] = 6 * E * I_z / L ** 2
        k_e[5, 5] = 4 * E * I_z / L
        k_e[5, 7] = -6 * E * I_z / L ** 2
        k_e[5, 11] = 2 * E * I_z / L
        k_e[7, 7] = 12 * E * I_z / L ** 3
        k_e[7, 11] = -6 * E * I_z / L ** 2
        k_e[11, 11] = 4 * E * I_z / L
        k_e[2, 2] = 12 * E * I_y / L ** 3
        k_e[2, 4] = -6 * E * I_y / L ** 2
        k_e[2, 8] = -12 * E * I_y / L ** 3
        k_e[2, 10] = -6 * E * I_y / L ** 2
        k_e[4, 4] = 4 * E * I_y / L
        k_e[4, 8] = 6 * E * I_y / L ** 2
        k_e[4, 10] = 2 * E * I_y / L
        k_e[8, 8] = 12 * E * I_y / L ** 3
        k_e[8, 10] = 6 * E * I_y / L ** 2
        k_e[10, 10] = 4 * E * I_y / L
        for i in range(12):
            for j in range(i + 1, 12):
                k_e[j, i] = k_e[i, j]
        return k_e
    for elem in elements:
        i = elem['node_i']
        j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        local_z = elem.get('local_z', None)
        node_i_coords = node_coords[i]
        node_j_coords = node_coords[j]
        (T, L, R) = compute_transformation_matrix(node_i_coords, node_j_coords, local_z)
        k_local = local_elastic_stiffness_matrix(L, E, A, I_y, I_z, J)
        k_global_elem = T.T @ k_local @ T
        dof_i = np.array([6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5])
        for (a, dof_a) in enumerate(dof_i):
            for (b, dof_b) in enumerate(dof_i):
                K_global[dof_a, dof_b] += k_global_elem[a, b]
    for (node_idx, loads) in nodal_loads.items():
        P_global[6 * node_idx:6 * node_idx + 6] = loads
    constrained_dofs = []
    for (node_idx, bc_flags) in boundary_conditions.items():
        for (dof_local, is_fixed) in enumerate(bc_flags):
            if is_fixed:
                dof_global = 6 * node_idx + dof_local
                constrained_dofs.append(dof_global)
    constrained_dofs = np.array(constrained_dofs, dtype=int)
    free_dofs = np.setdiff1d(np.arange(n_dof), constrained_dofs)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    P_f = P_global[free_dofs]
    try:
        u_free = np.linalg.solve(K_ff, P_f)
    except np.linalg.LinAlgError:
        raise ValueError('Stiffness matrix is singular; check boundary conditions.')
    u_global = np.zeros(n_dof)
    u_global[free_dofs] = u_free
    K_g_global = np.zeros((n_dof, n_dof))
    for elem in elements:
        i = elem['node_i']
        j = elem['node_j']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        local_z = elem.get('local_z', None)
        node_i_coords = node_coords[i]
        node_j_coords = node_coords[j]
        (T, L, R) = compute_transformation_matrix(node_i_coords, node_j_coords, local_z)
        dof_i_global = np.array([6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5])
        u_elem_global = u_global[dof_i_global]
        u_elem_local = T @ u_elem_global
        k_local = local_elastic_stiffness_matrix(L, elem['E'], A, I_y, I_z, J)
        f_local = k_local @ u_elem_local
        Fx2 = f_local[6]
        Mx2 = f_local[9]
        My1 = f_local[4]
        Mz1 = f_local[5]
        My2 = f_local[10]
        Mz2 = f_local[11]
        I_rho = J
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global_elem = T.T @ k_g_local @ T
        for (a, dof_a) in enumerate(dof_i_global):
            for (b, dof_b) in enumerate(dof_i_global):
                K_g_global[dof_a, dof_b] += k_g_global_elem[a, b]
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_g_ff = K_g_global[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_ff, -K_g_ff)
    except np.linalg.LinAlgError:
        raise ValueError('Generalized eigenproblem failed; check matrices.')
    positive_eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalue found.')
    lambda_cr = positive_eigenvalues[0]
    idx_min = np.where(eigenvalues == lambda_cr)[0][0]
    phi_free = eigenvectors[:, idx_min]
    phi_global = np.zeros(n_dof)
    phi_global[free_dofs] = phi_free
    return (lambda_cr, phi_global)