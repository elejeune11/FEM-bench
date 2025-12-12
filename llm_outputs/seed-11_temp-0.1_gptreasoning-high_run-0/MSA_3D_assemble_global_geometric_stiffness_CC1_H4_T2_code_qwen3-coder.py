def MSA_3D_assemble_global_geometric_stiffness_CC1_H4_T2(node_coords: np.ndarray, elements: Sequence[dict], u_global: np.ndarray) -> np.ndarray:
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    K_global = np.zeros((n_dof, n_dof))
    for elem in elements:
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        conn = elem['connectivity']
        (i_node, j_node) = (conn[0], conn[1])
        (xi, yi, zi) = node_coords[i_node]
        (xj, yj, zj) = node_coords[j_node]
        (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        (lx, ly, lz) = (dx / L, dy / L, dz / L)
        if 'local_z' in elem and elem['local_z'] is not None:
            local_z = np.array(elem['local_z'])
        else:
            global_z = np.array([0.0, 0.0, 1.0])
            beam_axis = np.array([lx, ly, lz])
            if np.abs(np.dot(beam_axis, global_z)) < 0.999:
                local_z = global_z
            else:
                local_z = np.array([0.0, 1.0, 0.0])
        local_x = np.array([lx, ly, lz])
        local_y = np.cross(local_z, local_x)
        local_y /= np.linalg.norm(local_y)
        local_z = np.cross(local_x, local_y)
        gamma = np.array([local_x, local_y, local_z])
        Gamma = np.zeros((12, 12))
        for i in range(4):
            idx = 3 * i
            Gamma[idx:idx + 3, idx:idx + 3] = gamma
        dofs = []
        for node in [i_node, j_node]:
            dofs.extend([6 * node, 6 * node + 1, 6 * node + 2, 6 * node + 3, 6 * node + 4, 6 * node + 5])
        u_elem_global = u_global[dofs]
        u_elem_local = Gamma @ u_elem_global
        k_local_elastic = np.zeros((12, 12))
        k_local_elastic[0, 0] = E * A / L
        k_local_elastic[0, 6] = -E * A / L
        k_local_elastic[6, 6] = E * A / L
        k_local_elastic[6, 0] = -E * A / L
        G = E / (2 * (1 + nu))
        k_local_elastic[3, 3] = G * J / L
        k_local_elastic[3, 9] = -G * J / L
        k_local_elastic[9, 9] = G * J / L
        k_local_elastic[9, 3] = -G * J / L
        k_local_elastic[1, 1] = 12 * E * I_z / L ** 3
        k_local_elastic[1, 5] = 6 * E * I_z / L ** 2
        k_local_elastic[1, 7] = -12 * E * I_z / L ** 3
        k_local_elastic[1, 11] = 6 * E * I_z / L ** 2
        k_local_elastic[5, 5] = 4 * E * I_z / L
        k_local_elastic[5, 7] = -6 * E * I_z / L ** 2
        k_local_elastic[5, 11] = 2 * E * I_z / L
        k_local_elastic[7, 7] = 12 * E * I_z / L ** 3
        k_local_elastic[7, 11] = -6 * E * I_z / L ** 2
        k_local_elastic[11, 11] = 4 * E * I_z / L
        k_local_elastic[2, 2] = 12 * E * I_y / L ** 3
        k_local_elastic[2, 4] = -6 * E * I_y / L ** 2
        k_local_elastic[2, 8] = -12 * E * I_y / L ** 3
        k_local_elastic[2, 10] = -6 * E * I_y / L ** 2
        k_local_elastic[4, 4] = 4 * E * I_y / L
        k_local_elastic[4, 8] = 6 * E * I_y / L ** 2
        k_local_elastic[4, 10] = 2 * E * I_y / L
        k_local_elastic[8, 8] = 12 * E * I_y / L ** 3
        k_local_elastic[8, 10] = 6 * E * I_y / L ** 2
        k_local_elastic[10, 10] = 4 * E * I_y / L
        k_local_elastic = k_local_elastic + k_local_elastic.T - np.diag(np.diag(k_local_elastic))
        f_local = k_local_elastic @ u_elem_local
        (Fx1, Fy1, Fz1, Mx1, My1, Mz1) = f_local[0:6]
        (Fx2, Fy2, Fz2, Mx2, My2, Mz2) = f_local[6:12]
        I_rho = I_y + I_z
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = Gamma.T @ k_g_local @ Gamma
        for (i_loc, i_glob) in enumerate(dofs):
            for (j_loc, j_glob) in enumerate(dofs):
                K_global[i_glob, j_glob] += k_g_global[i_loc, j_loc]
    return K_global