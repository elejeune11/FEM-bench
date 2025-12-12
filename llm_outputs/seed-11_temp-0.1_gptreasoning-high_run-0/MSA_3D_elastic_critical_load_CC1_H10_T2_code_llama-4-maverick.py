def MSA_3D_elastic_critical_load_CC1_H10_T2(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = len(node_coords)
    n_dofs = 6 * n_nodes
    K = np.zeros((n_dofs, n_dofs))
    P = np.zeros(n_dofs)
    for element in elements:
        (node_i, node_j) = (element['node_i'], element['node_j'])
        (x_i, y_i, z_i) = node_coords[node_i]
        (x_j, y_j, z_j) = node_coords[node_j]
        L = np.sqrt((x_j - x_i) ** 2 + (y_j - y_i) ** 2 + (z_j - z_i) ** 2)
        (E, nu, A, I_y, I_z, J) = (element['E'], element['nu'], element['A'], element['I_y'], element['I_z'], element['J'])
        G = E / (2 * (1 + nu))
        I_rho = I_y + I_z
        k_e_local = np.zeros((12, 12))
        k_e_local[0, 0] = k_e_local[6, 6] = E * A / L
        k_e_local[0, 6] = k_e_local[6, 0] = -E * A / L
        k_e_local[1, 1] = k_e_local[7, 7] = 12 * E * I_z / L ** 3
        k_e_local[1, 5] = k_e_local[5, 1] = 6 * E * I_z / L ** 2
        k_e_local[1, 7] = k_e_local[7, 1] = -12 * E * I_z / L ** 3
        k_e_local[1, 11] = k_e_local[11, 1] = 6 * E * I_z / L ** 2
        k_e_local[2, 2] = k_e_local[8, 8] = 12 * E * I_y / L ** 3
        k_e_local[2, 4] = k_e_local[4, 2] = -6 * E * I_y / L ** 2
        k_e_local[2, 8] = k_e_local[8, 2] = -12 * E * I_y / L ** 3
        k_e_local[2, 10] = k_e_local[10, 2] = -6 * E * I_y / L ** 2
        k_e_local[3, 3] = k_e_local[9, 9] = G * J / L
        k_e_local[3, 9] = k_e_local[9, 3] = -G * J / L
        k_e_local[4, 4] = k_e_local[10, 10] = 4 * E * I_y / L
        k_e_local[4, 8] = k_e_local[8, 4] = 6 * E * I_y / L ** 2
        k_e_local[4, 10] = k_e_local[10, 4] = 2 * E * I_y / L
        k_e_local[5, 5] = k_e_local[11, 11] = 4 * E * I_z / L
        k_e_local[5, 7] = k_e_local[7, 5] = -6 * E * I_z / L ** 2
        k_e_local[5, 11] = k_e_local[11, 5] = 2 * E * I_z / L
        k_e_local[7, 11] = k_e_local[11, 7] = -6 * E * I_z / L ** 2
        k_e_local[8, 10] = k_e_local[10, 8] = 6 * E * I_y / L ** 2
        k_e_local[10, 10] = 4 * E * I_y / L
        x_axis = np.array([x_j - x_i, y_j - y_i, z_j - z_i]) / L
        if 'local_z' in element:
            z_axis = np.array(element['local_z'])
            z_axis = z_axis / np.linalg.norm(z_axis)
            if np.dot(x_axis, z_axis) > 0.99:
                raise ValueError('local_z is parallel to the beam axis')
        elif np.linalg.norm(x_axis[:2]) < 1e-06:
            z_axis = np.array([0, 1, 0])
        else:
            z_axis = np.array([0, 0, 1])
        y_axis = np.cross(x_axis, z_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        T = np.zeros((12, 12))
        T[:3, :3] = T[3:6, 3:6] = T[6:9, 6:9] = T[9:, 9:] = np.column_stack((x_axis, y_axis, z_axis))
        k_e_global = T.T @ k_e_local @ T
        dofs_i = [6 * node_i, 6 * node_i + 1, 6 * node_i + 2, 6 * node_i + 3, 6 * node_i + 4, 6 * node_i + 5]
        dofs_j = [6 * node_j, 6 * node_j + 1, 6 * node_j + 2, 6 * node_j + 3, 6 * node_j + 4, 6 * node_j + 5]
        dofs = dofs_i + dofs_j
        for i in range(12):
            for j in range(12):
                K[dofs[i], dofs[j]] += k_e_global[i, j]
        if node_i in nodal_loads:
            P[dofs_i] += nodal_loads[node_i]
        if node_j in nodal_loads:
            P[dofs_j] += nodal_loads[node_j]
    free_dofs = np.ones(n_dofs, dtype=bool)
    for (node, bc) in boundary_conditions.items():
        dofs = [6 * node, 6 * node + 1, 6 * node + 2, 6 * node + 3, 6 * node + 4, 6 * node + 5]
        for i in range(6):
            if bc[i] == 1:
                free_dofs[dofs[i]] = False
    K_free = K[free_dofs, :][:, free_dofs]
    P_free = P[free_dofs]
    u_free = scipy.linalg.solve(K_free, P_free)
    u = np.zeros(n_dofs)
    u[free_dofs] = u_free
    K_g = np.zeros((n_dofs, n_dofs))
    for element in elements:
        (node_i, node_j) = (element['node_i'], element['node_j'])
        (x_i, y_i, z_i) = node_coords[node_i]
        (x_j, y_j, z_j) = node_coords[node_j]
        L = np.sqrt((x_j - x_i) ** 2 + (y_j - y_i) ** 2 + (z_j - z_i) ** 2)
        (A, I_rho) = (element['A'], element['I_y'] + element['I_z'])
        dofs_i = [6 * node_i, 6 * node_i + 1, 6 * node_i + 2, 6 * node_i + 3, 6 * node_i + 4, 6 * node_i + 5]
        dofs_j = [6 * node_j, 6 * node_j + 1, 6 * node_j + 2, 6 * node_j + 3, 6 * node_j + 4, 6 * node_j + 5]
        dofs = dofs_i + dofs_j
        u_element = u[dofs]
        x_axis = np.array([x_j - x_i, y_j - y_i, z_j - z_i]) / L
        if 'local_z' in element:
            z_axis = np.array(element['local_z'])
            z_axis = z_axis / np.linalg.norm(z_axis)
        elif np.linalg.norm(x_axis[:2]) < 1e-06:
            z_axis = np.array([0, 1, 0])
        else:
            z_axis = np.array([0, 0, 1])
        y_axis = np.cross(x_axis, z_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        T = np.zeros((12, 12))
        T[:3, :3] = T[3:6, 3:6] = T[6:9, 6:9] = T[9:, 9:] = np.column_stack((x_axis, y_axis, z_axis))
        u_local = T @ u_element
        (E, A) = (element['E'], element['A'])
        Fx2 = E * A / L * (u_local[6] - u_local[0])
        Mx2 = element['G'] * element['J'] / L * (u_local[9] - u_local[3])
        My1 = -2 * element['E'] * element['I_y'] / L * (2 * u_local[4] + u_local[10])
        Mz1 = 2 * element['E'] * element['I_z'] / L * (2 * u_local[5] + u_local[11])
        My2 = 2 * element['E'] * element['I_y'] / L * (u_local[4] + 2 * u_local[10])
        Mz2 = -2 * element['E'] * element['I_z'] / L * (u_local[5] + 2 * u_local[11])
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        k_g_global = T.T @ k_g_local @ T
        for i in range(12):
            for j in range(12):
                K_g[dofs[i], dofs[j]] += k_g_global[i, j]
    K_g_free = K_g[free_dofs, :][:, free_dofs]
    (eigenvalues, eigenvectors) = scipy.linalg.eig(K_free, -K_g_free)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalue found')
    elastic_critical_load_factor = np.min(positive_eigenvalues)
    mode_index = np.where(eigenvalues == elastic_critical_load_factor)[0][0]
    deformed_shape_vector = np.zeros(n_dofs)
    deformed_shape_vector[free_dofs] = eigenvectors[:, mode_index]
    return (elastic_critical_load_factor, deformed_shape_vector)