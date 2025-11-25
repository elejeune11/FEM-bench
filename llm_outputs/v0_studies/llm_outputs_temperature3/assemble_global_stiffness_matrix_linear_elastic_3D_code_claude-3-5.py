def assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements):
    n_nodes = len(node_coords)
    K = np.zeros((6 * n_nodes, 6 * n_nodes))
    for element in elements:
        (i, j) = (element['node_i'], element['node_j'])
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        E = element['E']
        nu = element['nu']
        A = element['A']
        Iy = element['I_y']
        Iz = element['I_z']
        J = element['J']
        ref_vec = element.get('local_z', None)
        if ref_vec is not None:
            ref_vec = np.array(ref_vec)
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ref_vec)
        k_global = Gamma.T @ k_local @ Gamma
        dofs_i = slice(6 * i, 6 * i + 6)
        dofs_j = slice(6 * j, 6 * j + 6)
        dofs = np.r_[dofs_i, dofs_j]
        K[np.ix_(dofs, dofs)] += k_global
    return K