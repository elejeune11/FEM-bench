def assemble_global_stiffness_matrix_linear_elastic_3D(elements, node_coords):
    n_nodes = len(node_coords)
    K = np.zeros((6 * n_nodes, 6 * n_nodes))
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        ref_vec = element.get('local_z', None)
        if ref_vec is not None:
            ref_vec = np.array(ref_vec)
        k_local = local_elastic_stiffness_matrix_3D_beam(element['E'], element['nu'], element['A'], np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2), element['I_y'], element['I_z'], element['J'])
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ref_vec)
        k_global = Gamma.T @ k_local @ Gamma
        dofs_i = slice(6 * node_i, 6 * node_i + 6)
        dofs_j = slice(6 * node_j, 6 * node_j + 6)
        dofs = np.r_[dofs_i, dofs_j]
        K[np.ix_(dofs, dofs)] += k_global
    return K