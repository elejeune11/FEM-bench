def assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements):
    n_nodes = node_coords.shape[0]
    K_global = np.zeros((6 * n_nodes, 6 * n_nodes))
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        E = element['E']
        nu = element['nu']
        A = element['A']
        I_y = element['I_y']
        I_z = element['I_z']
        J = element['J']
        local_z = element.get('local_z', None)
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, I_y, I_z, J)
        gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z)
        k_global = gamma.T @ k_local @ gamma
        dof_map = np.array([6 * node_i, 6 * node_i + 1, 6 * node_i + 2, 6 * node_i + 3, 6 * node_i + 4, 6 * node_i + 5, 6 * node_j, 6 * node_j + 1, 6 * node_j + 2, 6 * node_j + 3, 6 * node_j + 4, 6 * node_j + 5])
        for a in range(12):
            for b in range(12):
                K_global[dof_map[a], dof_map[b]] += k_global[a, b]
    return K_global