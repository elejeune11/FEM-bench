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
            if abs(x_local[2]) > 0.99:
                local_z_vec = np.array([0.0, 1.0, 0.0])
            else:
                local_z_vec = np.array([0.0, 0.0, 1.0])
        else:
            local_z_vec = np.array(local_z_vec, dtype=float)
            local_z_vec = local_z_vec / np.linalg.norm(local_z_vec)
        y_local = np.cross(local_z_vec, x_local)
        y_local = y_local / np.linalg.norm(y_local)
        z_local = np.cross(x_local, y_local)
        z_local = z_local / np.linalg.norm(z_local)
        R = np.array([x_local, y_local, z_local])
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        return (T, L)

    def local_elastic_stiffness_matrix(E, nu, A, I_y, I_z, J, L):
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
        (ni, nj) = (elem['node_i'], elem['node_j'])
        local_z = elem.get('local_z', None)
        (T, L) = get_transformation_matrix(node_coords[ni], node_coords[nj], local_z)
        k_local = local_elastic_stiffness_matrix(elem['E'], elem['nu'], elem['A'], elem['I_y'], elem['I_z'], elem['J'], L)
        k_global = T.T @ k_local @ T
        dofs = list(range(6 * ni, 6 * ni + 6)) + list(range(6 * nj, 6 * nj + 6))
        for (i_loc, i_glob) in enumerate(dofs):
            for (j_loc, j_glob) in enumerate(dofs):
                K_global[i_glob, j_glob] += k_global[i_loc, j_loc]
    P_global = np.zeros(n_dofs)
    for (node_idx, loads) in nodal_loads.items():
        for (i, load) in enumerate(loads):
            P_global[6 * node_idx + i] = load
    fixed_dofs = set()
    for (node_idx, bc) in boundary_conditions.items():
        for (i, is_fixed) in enumerate(bc):
            if is_fixed:
                fixed_dofs.add(6 * node_idx + i)
    free_dofs = sorted(set(range(n_dofs)) - fixed_dofs)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    P_f = P_global[free_dofs]
    u_f = np.linalg.solve(K_ff, P_f)
    u_global = np.zeros(n_dofs)
    for (i, dof) in enumerate(free_dofs):
        u_global[dof] = u_f[i]
    K_g_global = np.zeros((n_dofs, n_dofs))
    for elem in elements:
        (ni, nj) = (elem['node_i'], elem['node_j'])
        local_z = elem.get('local_z', None)
        (T, L) = get_transformation_matrix(node_coords[ni], node_coords[nj], local_z)
        dofs = list(range(6 * ni, 6 * ni + 6)) + list(range(6 * nj, 6 * nj + 6))
        u_elem_global = u_global[dofs]
        u_elem_local = T @ u_elem_global
        k_local = local_elastic_stiffness_matrix(elem['E'], elem['nu'], elem['A'], elem['I_y'], elem['I_z'], elem['J'], L)
        f_local = k_local @ u_elem_local
        Fx2 = f_local[6]
        Mx2 = f_local[9]
        My1 = f_local[4]
        Mz1 = f_local[5]
        My2 = f_local[10]
        Mz2 = f_local[11]
        I_rho = elem['I_y'] + elem['I_z']
        kg_local = local_geometric_stiffness_matrix_3D_beam(L, elem['A'], I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        kg_global = T.T @ kg_local @ T
        for (i_loc, i_glob) in enumerate(dofs):
            for (j_loc, j_glob) in enumerate(dofs):
                K_g_global[i_glob, j_glob] += kg_global[i_loc, j_loc]
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    Kg_ff = K_g_global[np.ix_(free_dofs, free_dofs)]
    (eigenvalues, eigenvectors) = scipy.linalg.eig(K_ff, -Kg_ff)
    real_eigenvalues = []
    real_eigenvectors = []
    for (i, ev) in enumerate(eigenvalues):
        if np.isreal(ev) or abs(ev.imag) < 1e-10 * abs(ev.real):
            real_ev = ev.real
            if real_ev > 1e-10:
                real_eigenvalues.append(real_ev)
                real_eigenvectors.append(eigenvectors[:, i].real)
    if not real_eigenvalues:
        raise ValueError('No positive eigenvalue found')
    min_idx = np.argmin(real_eigenvalues)
    lambda_cr = real_eigenvalues[min_idx]
    mode_f = real_eigenvectors[min_idx]
    mode_global = np.zeros(n_dofs)
    for (i, dof) in enumerate(free_dofs):
        mode_global[dof] = mode_f[i]
    return (lambda_cr, mode_global)