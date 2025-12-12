def MSA_3D_assemble_global_linear_elastic_stiffness_CC0_H2_T1(node_coords, elements):
    n_nodes = node_coords.shape[0]
    K_global = np.zeros((6 * n_nodes, 6 * n_nodes))
    for elem in elements:
        i = elem['node_i']
        j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        Iy = elem['I_y']
        Iz = elem['I_z']
        J = elem['J']
        local_z = elem.get('local_z', None)
        (x1, y1, z1) = node_coords[i]
        (x2, y2, z2) = node_coords[j]
        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        K_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        Gamma = beam_transformation_matrix_3D(x1, y1, z1, x2, y2, z2, local_z)
        K_global_local = Gamma.T @ K_local @ Gamma
        dof_indices = [6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5]
        for (idx1, global_idx1) in enumerate(dof_indices):
            for (idx2, global_idx2) in enumerate(dof_indices):
                K_global[global_idx1, global_idx2] += K_global_local[idx1, idx2]
    return K_global