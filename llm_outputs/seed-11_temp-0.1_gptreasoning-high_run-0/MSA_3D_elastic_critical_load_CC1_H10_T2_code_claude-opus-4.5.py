def MSA_3D_elastic_critical_load_CC1_H10_T2(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    """
    Perform linear (eigenvalue) buckling analysis for a 3D frame and return the
    elastic critical load factor and associated global buckling mode shape.
    """
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes

    def get_transformation_matrix(node_i_coords, node_j_coords, local_z_vec):
        """Compute 12x12 transformation matrix from local to global coordinates."""
        dx = node_j_coords - node_i_coords
        L = np.linalg.norm(dx)
        x_local = dx / L
        if local_z_vec is None:
            global_z = np.array([0.0, 0.0, 1.0])
            global_y = np.array([0.0, 1.0, 0.0])
            if np.abs(np.abs(np.dot(x_local, global_z)) - 1.0) < 1e-06:
                z_local = global_y
            else:
                z_local = global_z
        else:
            z_local = np.array(local_z_vec, dtype=float)
        y_local = np.cross(z_local, x_local)
        y_local = y_local / np.linalg.norm(y_local)
        z_local = np.cross(x_local, y_local)
        z_local = z_local / np.linalg.norm(z_local)
        R = np.array([x_local, y_local, z_local]).T
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        return (T, L, R)

    def local_elastic_stiffness_matrix(L, E, A, I_y, I_z, J, nu):
        """Compute 12x12 local elastic stiffness matrix for 3D beam."""
        G = E / (2 * (1 + nu))
        k = np.zeros((12, 12))
        k[0, 0] = E * A / L
        k[0, 6] = -E * A / L
        k[6, 0] = -E * A / L
        k[6, 6] = E * A / L
        k[3, 3] = G * J / L
        k[3, 9] = -G * J / L
        k[9, 3] = -G * J / L
        k[9, 9] = G * J / L
        k[1, 1] = 12 * E * I_z / L ** 3
        k[1, 5] = 6 * E * I_z / L ** 2
        k[1, 7] = -12 * E * I_z / L ** 3
        k[1, 11] = 6 * E * I_z / L ** 2
        k[5, 1] = 6 * E * I_z / L ** 2
        k[5, 5] = 4 * E * I_z / L
        k[5, 7] = -6 * E * I_z / L ** 2
        k[5, 11] = 2 * E * I_z / L
        k[7, 1] = -12 * E * I_z / L ** 3
        k[7, 5] = -6 * E * I_z / L ** 2
        k[7, 7] = 12 * E * I_z / L ** 3
        k[7, 11] = -6 * E * I_z / L ** 2
        k[11, 1] = 6 * E * I_z / L ** 2
        k[11, 5] = 2 * E * I_z / L
        k[11, 7] = -6 * E * I_z / L ** 2
        k[11, 11] = 4 * E * I_z / L
        k[2, 2] = 12 * E * I_y / L ** 3
        k[2, 4] = -6 * E * I_y / L ** 2
        k[2, 8] = -12 * E * I_y / L ** 3
        k[2, 10] = -6 * E * I_y / L ** 2
        k[4, 2] = -6 * E * I_y / L ** 2
        k[4, 4] = 4 * E * I_y / L
        k[4, 8] = 6 * E * I_y / L ** 2
        k[4, 10] = 2 * E * I_y / L
        k[8, 2] = -12 * E * I_y / L ** 3
        k[8, 4] = 6 * E * I_y / L ** 2
        k[8, 8] = 12 * E * I_y / L ** 3
        k[8, 10] = 6 * E * I_y / L ** 2
        k[10, 2] = -6 * E * I_y / L ** 2
        k[10, 4] = 2 * E * I_y / L
        k[10, 8] = 6 * E * I_y / L ** 2
        k[10, 10] = 4 * E * I_y / L
        return k
    K_global = np.zeros((n_dofs, n_dofs))
    for elem in elements:
        ni = elem['node_i']
        nj = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        local_z = elem.get('local_z', None)
        node_i_coords = node_coords[ni]
        node_j_coords = node_coords[nj]
        (T, L, R) = get_transformation_matrix(node_i_coords, node_j_coords, local_z)
        k_local = local_elastic_stiffness_matrix(L, E, A, I_y, I_z, J, nu)
        k_global = T @ k_local @ T.T
        dofs = np.array([6 * ni, 6 * ni + 1, 6 * ni + 2, 6 * ni + 3, 6 * ni + 4, 6 * ni + 5, 6 * nj, 6 * nj + 1, 6 * nj + 2, 6 * nj + 3, 6 * nj + 4, 6 * nj + 5])
        for i in range(12):
            for j in range(12):
                K_global[dofs[i], dofs[j]] += k_global[i, j]
    P_global = np.zeros(n_dofs)
    for (node_idx, loads) in nodal_loads.items():
        for i in range(6):
            P_global[6 * node_idx + i] = loads[i]
    fixed_dofs = set()
    for (node_idx, bc) in boundary_conditions.items():
        for i in range(6):
            if bc[i] == 1:
                fixed_dofs.add(6 * node_idx + i)
    all_dofs = set(range(n_dofs))
    free_dofs = sorted(all_dofs - fixed_dofs)
    free_dofs = np.array(free_dofs)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    P_f = P_global[free_dofs]
    u_f = np.linalg.solve(K_ff, P_f)
    u_global = np.zeros(n_dofs)
    u_global[free_dofs] = u_f
    K_g_global = np.zeros((n_dofs, n_dofs))
    for elem in elements:
        ni = elem['node_i']
        nj = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        local_z = elem.get('local_z', None)
        node_i_coords = node_coords[ni]
        node_j_coords = node_coords[nj]
        (T, L, R) = get_transformation_matrix(node_i_coords, node_j_coords, local_z)
        k_local = local_elastic_stiffness_matrix(L, E, A, I_y, I_z, J, nu)
        dofs = np.array([6 * ni, 6 * ni + 1, 6 * ni + 2, 6 * ni + 3, 6 * ni + 4, 6 * ni + 5, 6 * nj, 6 * nj + 1, 6 * nj + 2, 6 * nj + 3, 6 * nj + 4, 6 * nj + 5])
        u_elem_global = u_global[dofs]
        u_elem_local = T.T @ u_elem_global
        f_local = k_local @ u_elem_local
        Fx2 = f_local[6]
        Mx2 = f_local[9]
        My1 = f_local[4]
        Mz1 = f_local[5]
        My2 = f_local[10]
        Mz2 = f_local[11]
        I_rho = I_y + I_z
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = T @ k_g_local @ T.T
        for i in range(12):
            for j in range(12):
                K_g_global[dofs[i], dofs[j]] += k_g_global[i, j]
    K_g_ff = K_g_global[np.ix_(free_dofs, free_dofs)]
    (eigenvalues, eigenvectors) = scipy.linalg.eig(K_ff, -K_g_ff)
    real_mask = np.abs(eigenvalues.imag) < 1e-06 * np.abs(eigenvalues.real + 1e-30)
    eigenvalues_real = eigenvalues.real
    positive_mask = (eigenvalues_real > 1e-10) & real_mask
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalue found')
    positive_eigenvalues = eigenvalues_real[positive_mask]
    positive_eigenvectors = eigenvectors[:, positive_mask]
    min_idx = np.argmin(positive_eigenvalues)
    lambda_cr = positive_eigenvalues[min_idx]
    phi_f = positive_eigenvectors[:, min_idx].real
    phi_global = np.zeros(n_dofs)
    phi_global[free_dofs] = phi_f
    return (lambda_cr, phi_global)