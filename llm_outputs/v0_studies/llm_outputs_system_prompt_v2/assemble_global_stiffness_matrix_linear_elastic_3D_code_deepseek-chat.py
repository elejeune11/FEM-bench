def assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements):
    n_nodes = node_coords.shape[0]
    K_global = np.zeros((6 * n_nodes, 6 * n_nodes))
    for element in elements:
        i = element['node_i']
        j = element['node_j']
        (x1, y1, z1) = node_coords[i]
        (x2, y2, z2) = node_coords[j]
        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        local_z = element.get('local_z', None)
        if local_z is not None:
            local_z = np.array(local_z)
        Gamma = beam_transformation_matrix_3D(x1, y1, z1, x2, y2, z2, local_z)
        k_local = local_elastic_stiffness_matrix_3D_beam(element['E'], element['nu'], element['A'], L, element['I_y'], element['I_z'], element['J'])
        k_global_element = Gamma.T @ k_local @ Gamma
        dofs_i = slice(6 * i, 6 * i + 6)
        dofs_j = slice(6 * j, 6 * j + 6)
        K_global[dofs_i, dofs_i] += k_global_element[0:6, 0:6]
        K_global[dofs_i, dofs_j] += k_global_element[0:6, 6:12]
        K_global[dofs_j, dofs_i] += k_global_element[6:12, 0:6]
        K_global[dofs_j, dofs_j] += k_global_element[6:12, 6:12]
    return K_global