def MSA_3D_assemble_global_geometric_stiffness_CC1_H4_T3(node_coords: np.ndarray, elements: Sequence[dict], u_global: np.ndarray) -> np.ndarray:
    n_nodes = len(node_coords)
    K_g_global = np.zeros((6 * n_nodes, 6 * n_nodes))
    for element in elements:
        (node_i, node_j) = (element['node_i'], element['node_j'])
        (E, nu, A, I_y, I_z, J) = (element['E'], element['nu'], element['A'], element['I_y'], element['I_z'], element['J'])
        local_z = element.get('local_z', None)
        (x_i, y_i, z_i) = node_coords[node_i]
        (x_j, y_j, z_j) = node_coords[node_j]
        L = np.sqrt((x_j - x_i) ** 2 + (y_j - y_i) ** 2 + (z_j - z_i) ** 2)
        axis = np.array([x_j - x_i, y_j - y_i, z_j - z_i]) / L
        if local_z is None:
            if np.abs(axis[2]) < 0.9:
                local_z = np.array([0, 0, 1])
            else:
                local_z = np.array([0, 1, 0])
        else:
            local_z = np.array(local_z)
        local_z = local_z / np.linalg.norm(local_z)
        if np.abs(np.dot(local_z, axis)) > 0.9:
            raise ValueError('local_z is parallel to the beam axis')
        local_x = axis
        local_y = np.cross(local_z, local_x)
        local_y = local_y / np.linalg.norm(local_y)
        local_z = np.cross(local_x, local_y)
        T = np.array([local_x, local_y, local_z]).T
        u_local = np.zeros(12)
        u_local[:6] = T.T @ np.array([u_global[6 * node_i + i] for i in range(3)] + [u_global[6 * node_i + i + 3] for i in range(3)])
        u_local[6:] = T.T @ np.array([u_global[6 * node_j + i] for i in range(3)] + [u_global[6 * node_j + i + 3] for i in range(3)])
        (Fx_i, Fy_i, Fz_i, Mx_i, My_i, Mz_i) = (0, 0, 0, 0, 0, 0)
        (Fx_j, Fy_j, Fz_j, Mx_j, My_j, Mz_j) = (0, 0, 0, 0, 0, 0)
        EA = E * A
        EI_y = E * I_y
        EI_z = E * I_z
        GJ = E / (2 * (1 + nu)) * J
        (u1, v1, w1, theta_x1, theta_y1, theta_z1) = u_local[:6]
        (u2, v2, w2, theta_x2, theta_y2, theta_z2) = u_local[6:]
        Fx_i = EA / L * (u2 - u1)
        Fy_i = 12 * EI_z / L ** 3 * (v2 - v1) - 6 * EI_z / L ** 2 * (theta_z2 + theta_z1)
        Fz_i = 12 * EI_y / L ** 3 * (w2 - w1) + 6 * EI_y / L ** 2 * (theta_y2 + theta_y1)
        Mx_i = GJ / L * (theta_x2 - theta_x1)
        My_i = -6 * EI_y / L ** 2 * (w2 - w1) - 4 * EI_y / L * theta_y1 - 2 * EI_y / L * theta_y2
        Mz_i = 6 * EI_z / L ** 2 * (v2 - v1) + 4 * EI_z / L * theta_z1 + 2 * EI_z / L * theta_z2
        Fx_j = -Fx_i
        Fy_j = -Fy_i
        Fz_j = -Fz_i
        Mx_j = -Mx_i
        My_j = 6 * EI_y / L ** 2 * (w2 - w1) + 2 * EI_y / L * theta_y1 + 4 * EI_y / L * theta_y2
        Mz_j = -6 * EI_z / L ** 2 * (v2 - v1) - 2 * EI_z / L * theta_z1 - 4 * EI_z / L * theta_z2
        k_g_local = np.zeros((12, 12))
        k_g_local[0, 0] = 0
        k_g_local[0, 6] = 0
        k_g_local[1, 1] = 6 / 5 * Fx_i / L
        k_g_local[1, 5] = -Fx_i / 10
        k_g_local[1, 7] = -(6 / 5) * Fx_i / L
        k_g_local[1, 11] = -Fx_i / 10
        k_g_local[2, 2] = 6 / 5 * Fx_i / L
        k_g_local[2, 4] = Fx_i / 10
        k_g_local[2, 8] = -(6 / 5) * Fx_i / L
        k_g_local[2, 10] = Fx_i / 10
        k_g_local[3, 3] = (My_i * theta_y1 + Mz_i * theta_z1 + My_j * theta_y2 + Mz_j * theta_z2) / L
        k_g_local[3, 9] = -k_g_local[3, 3]
        k_g_local[4, 2] = Fx_i / 10
        k_g_local[4, 4] = 2 / 15 * Fx_i * L
        k_g_local[4, 8] = -Fx_i / 10
        k_g_local[4, 10] = -(1 / 30) * Fx_i * L
        k_g_local[5, 1] = -Fx_i / 10
        k_g_local[5, 5] = 2 / 15 * Fx_i * L
        k_g_local[5, 7] = Fx_i / 10
        k_g_local[5, 11] = -(1 / 30) * Fx_i * L
        k_g_local[6, 0] = 0
        k_g_local[6, 6] = 0
        k_g_local[7, 1] = -(6 / 5) * Fx_i / L
        k_g_local[7, 5] = Fx_i / 10
        k_g_local[7, 7] = 6 / 5 * Fx_i / L
        k_g_local[7, 11] = Fx_i / 10
        k_g_local[8, 2] = -(6 / 5) * Fx_i / L
        k_g_local[8, 4] = -Fx_i / 10
        k_g_local[8, 8] = 6 / 5 * Fx_i / L
        k_g_local[8, 10] = -Fx_i / 10
        k_g_local[9, 3] = -k_g_local[3, 3]
        k_g_local[9, 9] = k_g_local[3, 3]
        k_g_local[10, 2] = Fx_i / 10
        k_g_local[10, 4] = -(1 / 30) * Fx_i * L
        k_g_local[10, 8] = -Fx_i / 10
        k_g_local[10, 10] = 2 / 15 * Fx_i * L
        k_g_local[11, 1] = -Fx_i / 10
        k_g_local[11, 5] = -(1 / 30) * Fx_i * L
        k_g_local[11, 7] = Fx_i / 10
        k_g_local[11, 11] = 2 / 15 * Fx_i * L
        k_g_local += k_g_local.T - np.diag(np.diag(k_g_local))
        Gamma = np.block([[T, np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))], [np.zeros((3, 3)), T, np.zeros((3, 3)), np.zeros((3, 3))], [np.zeros((3, 3)), np.zeros((3, 3)), T, np.zeros((3, 3))], [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), T]])
        k_g_global = Gamma @ k_g_local @ Gamma.T
        dofs = [6 * node_i + i for i in range(6)] + [6 * node_j + i for i in range(6)]
        for (i, dof_i) in enumerate(dofs):
            for (j, dof_j) in enumerate(dofs):
                K_g_global[dof_i, dof_j] += k_g_global[i, j]
    return K_g_global