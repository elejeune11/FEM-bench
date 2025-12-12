def MSA_3D_elastic_critical_load_CC1_H10_T3(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    K = np.zeros((len(node_coords) * 6, len(node_coords) * 6))
    for element in elements:
        (node_i, node_j) = (element['node_i'], element['node_j'])
        (E, nu, A, I_y, I_z, J) = (element['E'], element['nu'], element['A'], element['I_y'], element['I_z'], element['J'])
        local_z = element.get('local_z', None)
        if local_z is None:
            (x_i, y_i, z_i) = node_coords[node_i]
            (x_j, y_j, z_j) = node_coords[node_j]
            beam_axis = np.array([x_j - x_i, y_j - y_i, z_j - z_i])
            if np.allclose(beam_axis, [0, 0, 1]) or np.allclose(beam_axis, [0, 0, -1]):
                local_z = [0, 1, 0]
            else:
                local_z = [0, 0, 1]
        k = _frame3D_stiffness(E, nu, A, I_y, I_z, J, node_coords[node_i], node_coords[node_j], local_z)
        _assemble_global_stiffness(K, k, node_i, node_j)
    P = np.zeros(len(node_coords) * 6)
    for (node, loads) in nodal_loads.items():
        P[6 * node:6 * (node + 1)] = loads
    free_dofs = np.ones(len(node_coords) * 6, dtype=bool)
    for (node, bc) in boundary_conditions.items():
        free_dofs[6 * node:6 * (node + 1)] &= [not b for b in bc]
    K_free = K[free_dofs, :][:, free_dofs]
    P_free = P[free_dofs]
    u_free = scipy.linalg.solve(K_free, P_free)
    u = np.zeros(len(node_coords) * 6)
    u[free_dofs] = u_free
    K_g = np.zeros((len(node_coords) * 6, len(node_coords) * 6))
    for element in elements:
        (node_i, node_j) = (element['node_i'], element['node_j'])
        (E, nu, A, I_y, I_z, J) = (element['E'], element['nu'], element['A'], element['I_y'], element['I_z'], element['J'])
        local_z = element.get('local_z', None)
        if local_z is None:
            (x_i, y_i, z_i) = node_coords[node_i]
            (x_j, y_j, z_j) = node_coords[node_j]
            beam_axis = np.array([x_j - x_i, y_j - y_i, z_j - z_i])
            if np.allclose(beam_axis, [0, 0, 1]) or np.allclose(beam_axis, [0, 0, -1]):
                local_z = [0, 1, 0]
            else:
                local_z = [0, 0, 1]
        internal_forces = _frame3D_internal_forces(E, nu, A, I_y, I_z, J, node_coords[node_i], node_coords[node_j], local_z, u)
        k_g = _frame3D_geometric_stiffness(E, nu, A, I_y, I_z, J, node_coords[node_i], node_coords[node_j], local_z, internal_forces)
        _assemble_global_stiffness(K_g, k_g, node_i, node_j)
    K_g_free = K_g[free_dofs, :][:, free_dofs]
    (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_free, -K_g_free)
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalue found')
    elastic_critical_load_factor = np.min(positive_eigenvalues)
    mode_shape_free = eigenvectors[:, np.argmin(eigenvalues >= 0)]
    deformed_shape_vector = np.zeros(len(node_coords) * 6)
    deformed_shape_vector[free_dofs] = mode_shape_free
    return (elastic_critical_load_factor, deformed_shape_vector)