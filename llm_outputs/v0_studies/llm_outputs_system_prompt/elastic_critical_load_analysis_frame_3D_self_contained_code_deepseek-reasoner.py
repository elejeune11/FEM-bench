def elastic_critical_load_analysis_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K_global = np.zeros((n_dofs, n_dofs))
    P_global = np.zeros(n_dofs)
    for element in elements:
        i = element['node_i']
        j = element['node_j']
        L = np.linalg.norm(node_coords[j] - node_coords[i])
        E = element['E']
        A = element['A']
        Iy = element['Iy']
        Iz = element['Iz']
        J = element['J']
        nu = element['nu']
        G = E / (2 * (1 + nu))
        k_local = np.zeros((12, 12))
        k_local[0, 0] = E * A / L
        k_local[0, 6] = -E * A / L
        k_local[6, 0] = -E * A / L
        k_local[6, 6] = E * A / L
        k_local[1, 1] = 12 * E * Iz / L ** 3
        k_local[1, 5] = 6 * E * Iz / L ** 2
        k_local[1, 7] = -12 * E * Iz / L ** 3
        k_local[1, 11] = 6 * E * Iz / L ** 2
        k_local[5, 1] = 6 * E * Iz / L ** 2
        k_local[5, 5] = 4 * E * Iz / L
        k_local[5, 7] = -6 * E * Iz / L ** 2
        k_local[5, 11] = 2 * E * Iz / L
        k_local[7, 1] = -12 * E * Iz / L ** 3
        k_local[7, 5] = -6 * E * Iz / L ** 2
        k_local[7, 7] = 12 * E * Iz / L ** 3
        k_local[7, 11] = -6 * E * Iz / L ** 2
        k_local[11, 1] = 6 * E * Iz / L ** 2
        k_local[11, 5] = 2 * E * Iz / L
        k_local[11, 7] = -6 * E * Iz / L ** 2
        k_local[11, 11] = 4 * E * Iz / L
        k_local[2, 2] = 12 * E * Iy / L ** 3
        k_local[2, 4] = -6 * E * Iy / L ** 2
        k_local[2, 8] = -12 * E * Iy / L ** 3
        k_local[2, 10] = -6 * E * Iy / L ** 2
        k_local[4, 2] = -6 * E * Iy / L ** 2
        k_local[4, 4] = 4 * E * Iy / L
        k_local[4, 8] = 6 * E * Iy / L ** 2
        k_local[4, 10] = 2 * E * Iy / L
        k_local[8, 2] = -12 * E * Iy / L ** 3
        k_local[8, 4] = 6 * E * Iy / L ** 2
        k_local[8, 8] = 12 * E * Iy / L ** 3
        k_local[8, 10] = 6 * E * Iy / L ** 2
        k_local[10, 2] = -6 * E * Iy / L ** 2
        k_local[10, 4] = 2 * E * Iy / L
        k_local[10, 8] = 6 * E * Iy / L ** 2
        k_local[10, 10] = 4 * E * Iy / L
        k_local[3, 3] = G * J / L
        k_local[3, 9] = -G * J / L
        k_local[9, 3] = -G * J / L
        k_local[9, 9] = G * J / L
        vec_x = node_coords[j] - node_coords[i]
        vec_x = vec_x / np.linalg.norm(vec_x)
        if element['local_z'] is not None:
            vec_z = np.array(element['local_z'])
            vec_z = vec_z / np.linalg.norm(vec_z)
            vec_y = np.cross(vec_z, vec_x)
            vec_y = vec_y / np.linalg.norm(vec_y)
            vec_z = np.cross(vec_x, vec_y)
        else:
            if abs(vec_x[2]) > 0.9:
                vec_y = np.array([0.0, 1.0, 0.0])
            else:
                vec_y = np.array([0.0, 0.0, 1.0])
            vec_z = np.cross(vec_x, vec_y)
            vec_z = vec_z / np.linalg.norm(vec_z)
            vec_y = np.cross(vec_z, vec_x)
        R = np.column_stack((vec_x, vec_y, vec_z))
        T = np.zeros((12, 12))
        for block in range(4):
            start = 3 * block
            end = 3 * (block + 1)
            T[start:end, start:end] = R
        k_global_elem = T.T @ k_local @ T
        dofs_i = slice(6 * i, 6 * i + 6)
        dofs_j = slice(6 * j, 6 * j + 6)
        K_global[dofs_i, dofs_i] += k_global_elem[0:6, 0:6]
        K_global[dofs_i, dofs_j] += k_global_elem[0:6, 6:12]
        K_global[dofs_j, dofs_i] += k_global_elem[6:12, 0:6]
        K_global[dofs_j, dofs_j] += k_global_elem[6:12, 6:12]
    for (node, loads) in nodal_loads.items():
        start_dof = 6 * node
        P_global[start_dof:start_dof + 6] = loads
    fixed_dofs = set()
    for (node, bc) in boundary_conditions.items():
        if all((isinstance(x, bool) for x in bc)):
            for (dof_idx, is_fixed) in enumerate(bc):
                if is_fixed:
                    fixed_dofs.add(6 * node + dof_idx)
        else:
            for dof_idx in bc:
                fixed_dofs.add(6 * node + dof_idx)
    free_dofs = sorted(set(range(n_dofs)) - fixed_dofs)
    K_free = K_global[np.ix_(free_dofs, free_dofs)]
    P_free = P_global[free_dofs]
    u_free = scipy.linalg.solve(K_free, P_free, assume_a='sym')
    u_global = np.zeros(n_dofs)
    u_global[free_dofs] = u_free
    K_geo_global = np.zeros((n_dofs, n_dofs))
    for element in elements:
        i = element['node_i']
        j = element['node_j']
        L = np.linalg.norm(node_coords[j] - node_coords[i])
        E = element['E']
        A = element['A']
        I_rho = element['I_rho']
        u_elem = np.concatenate([u_global[6 * i:6 * i + 6], u_global[6 * j:6 * j + 6]])
        vec_x = node_coords[j] - node_coords[i]
        vec_x = vec_x / np.linalg.norm(vec_x)
        if element['local_z'] is not None:
            vec_z = np.array(element['local_z'])
            vec_z = vec_z / np.linalg.norm(vec_z)
            vec_y = np.cross(vec_z, vec_x)
            vec_y = vec_y / np.linalg.norm(vec_y)
            vec_z = np.cross(vec_x, vec_y)
        else:
            if abs(vec_x[2]) > 0.9:
                vec_y = np.array([0.0, 1.0, 0.0])
            else:
                vec_y = np.array([0.0, 0.0, 1.0])
            vec_z = np.cross(vec_x, vec_y)
            vec_z = vec_z / np.linalg.norm(vec_z)
            vec_y = np.cross(vec_z, vec_x)
        R = np.column_stack((vec_x, vec_y, vec_z))
        T = np.zeros((12, 12))
        for block in range(4):
            start = 3 * block
            end = 3 * (block + 1)
            T[start:end, start:end] = R
        u_local = T @ u_elem
        axial_force = E * A * (u_local[6] - u_local[0]) / L
        k_geo_local = np.zeros((12, 12))
        k_geo_local[1, 1] = 6 / (5 * L)
        k_geo_local[1, 5] = 1 / 10
        k_geo_local[1, 7] = -6 / (5 * L)
        k_geo_local[1, 11] = 1 / 10
        k_geo_local[5, 1] = 1 / 10
        k_geo_local[5, 5] = 2 * L / 15
        k_geo_local[5, 7] = -1 / 10
        k_geo_local[5, 11] = -L / 30
        k_geo_local[7, 1] = -6 / (5 * L)
        k_geo_local[7, 5] = -1 / 10
        k_geo_local[7, 7] = 6 / (5 * L)
        k_geo_local[7, 11] = -1 / 10
        k_geo_local[11, 1] = 1 / 10
        k_geo_local[11, 5] = -L / 30
        k_geo_local[11, 7] = -1 / 10
        k_geo_local[11, 11] = 2 * L / 15
        k_geo_local[2, 2] = 6 / (5 * L)
        k_geo_local[2, 4] = -1 / 10
        k_geo_local[2, 8] = -6 / (5 * L)
        k_geo_local[2, 10] = -1 / 10
        k_geo_local[4, 2] = -1 / 10
        k_geo_local[4, 4] = 2 * L / 15
        k_geo_local[4, 8] = 1 / 10
        k_geo_local[4, 10] = -L / 30
        k_geo_local[8, 2] = -6 / (5 * L)
        k_geo_local[8, 4] = 1 / 10
        k_geo_local[8, 8] = 6 / (5 * L)
        k_geo_local[8, 10] = 1 / 10
        k_geo_local[10, 2] = -1 / 10
        k_geo_local[10, 4] = -L / 30
        k_geo_local[10, 8] = 1 / 10
        k_geo_local[10, 10] = 2 * L / 15
        k_geo_local[3, 3] = I_rho / (A * L)
        k_geo_local[3, 9] = -I_rho / (A * L)
        k_geo_local[9, 3] = -I_rho / (A * L)
        k_geo_local[9, 9] = I_rho / (A * L)
        k_geo_local *= axial_force
        k_geo_global_elem = T.T @ k_geo_local @ T
        dofs_i = slice(6 * i, 6 * i + 6)
        dofs_j = slice(6 * j, 6 * j + 6)
        K_geo_global[dofs_i, dofs_i] += k_geo_global_elem[0:6, 0:6]
        K_geo_global[dofs_i, dofs_j] += k_geo_global_elem[0:6, 6:12]
        K_geo_global[dofs_j, dofs_i] += k_geo_global_elem[6:12, 0:6]
        K_geo_global[dofs_j, dofs_j] += k_geo_global_elem[6:12, 6:12]
    K_free_geo = K_geo_global[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eig(K_free, -K_free_geo)
    except:
        (eigenvalues, eigenvectors) = scipy.linalg.eig(K_free, -K_free_geo, check_finite=False)
    real_eigenvalues = np.real(eigenvalues)
    positive_eigenvalues = real_eigenvalues[real_eigenvalues > 0]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalues found')
    min_positive_idx = np.argmin(positive_eigenvalues)
    critical_load_factor = positive_eigenvalues[min_positive_idx]
    original_idx = np.where(real_eigenvalues == critical_load_factor)[0][0]
    mode_free = np.real(eigenvectors[:, original_idx])
    deformed_shape = np.zeros(n_dofs)
    deformed_shape[free_dofs] = mode_free
    return (float(critical_load_factor), deformed_shape)