def assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements):
    n_nodes = len(node_coords)
    K = np.zeros((6 * n_nodes, 6 * n_nodes))
    for element in elements:
        (i, j) = (element['node_i'], element['node_j'])
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        local_z = element.get('local_z', None)
        if local_z is not None:
            local_z = np.array(local_z)
        dx = xj - xi
        dy = yj - yi
        dz = zj - zi
        L = np.sqrt(dx * dx + dy * dy + dz * dz)
        k_local = local_elastic_stiffness_matrix_3D_beam(element['E'], element['nu'], element['A'], L, element['I_y'], element['I_z'], element['J'])
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z)
        k_global = Gamma.T @ k_local @ Gamma
        dofs_i = slice(6 * i, 6 * i + 6)
        dofs_j = slice(6 * j, 6 * j + 6)
        dofs = [*range(6 * i, 6 * i + 6), *range(6 * j, 6 * j + 6)]
        for p in range(12):
            for q in range(12):
                K[dofs[p], dofs[q]] += k_global[p, q]
    return K