def assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements):
    n_nodes = node_coords.shape[0]
    K = np.zeros((6 * n_nodes, 6 * n_nodes))
    for element in elements:
        i = element['node_i']
        j = element['node_j']
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        E = element['E']
        nu = element['nu']
        A = element['A']
        Iy = element['I_y']
        Iz = element['I_z']
        J = element['J']
        local_z = element.get('local_z', None)
        K_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z)
        K_global_element = Gamma.T @ K_local @ Gamma
        dofs_i = slice(6 * i, 6 * i + 6)
        dofs_j = slice(6 * j, 6 * j + 6)
        element_dofs = np.r_[dofs_i, dofs_j]
        K[np.ix_(element_dofs, element_dofs)] += K_global_element
    return K