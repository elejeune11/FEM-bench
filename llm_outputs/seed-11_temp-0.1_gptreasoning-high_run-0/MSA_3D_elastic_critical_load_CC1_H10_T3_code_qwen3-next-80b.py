def MSA_3D_elastic_critical_load_CC1_H10_T3(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        local_z = elem.get('local_z')
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        dx = (xj - xi) / L
        dy = (yj - yi) / L
        dz = (zj - zi) / L
        if local_z is None:
            if abs(dz) > 0.999:
                local_z = np.array([0.0, 1.0, 0.0])
            else:
                local_z = np.array([0.0, 0.0, 1.0])
        local_z = local_z / np.linalg.norm(local_z)
        local_x = np.array([dx, dy, dz])
        local_y = np.cross(local_z, local_x)
        local_y = local_y / np.linalg.norm(local_y)
        local_z = np.cross(local_x, local_y)
        T = np.zeros((12, 12))
        for i in range(3):
            T[i, i] = local_x[i]
            T[i, i + 3] = local_y[i]
            T[i, i + 6] = local_z[i]
            T[i + 3, i] = local_x[i]
            T[i + 3, i + 3] = local_y[i]
            T[i + 3, i + 6] = local_z[i]
            T[i + 6, i] = local_x[i]
            T[i + 6, i + 3] = local_y[i]
            T[i + 6, i + 6] = local_z[i]
            T[i + 9, i] = local_x[i]
            T[i + 9, i + 3] = local_y[i]
            T[i + 9, i + 6] = local_z[i]
        k_local = np.zeros((12, 12))
        k_local[0, 0] = E * A / L
        k_local[0, 6] = -E * A / L
        k_local[6, 0] = -E * A / L
        k_local[6, 6] = E * A / L
        k_local[1, 1] = 12 * E * I_z / L ** 3
        k_local[1, 5] = 6 * E * I_z / L ** 2
        k_local[1, 7] = -12 * E * I_z / L ** 3
        k_local[1, 11] = 6 * E * I_z / L ** 2
        k_local[5, 1] = 6 * E * I_z / L ** 2
        k_local[5, 5] = 4 * E * I_z / L
        k_local[5, 7] = -6 * E * I_z / L ** 2
        k_local[5, 11] = 2 * E * I_z / L
        k_local[7, 1] = -12 * E * I_z / L ** 3
        k_local[7, 5] = -6 * E * I_z / L ** 2
        k_local[7, 7] = 12 * E * I_z / L ** 3
        k_local[7, 11] = -6 * E * I_z / L ** 2
        k_local[11, 1] = 6 * E * I_z / L ** 2
        k_local[11, 5] = 2 * E * I_z / L
        k_local[11, 7] = -6 * E * I_z / L ** 2
        k_local[11, 11] = 4 * E * I_z / L
        k_local[2, 2] = 12 * E * I_y / L ** 3
        k_local[2, 4] = -6 * E * I_y / L ** 2
        k_local[2, 8] = -12 * E * I_y / L ** 3
        k_local[2, 10] = -6 * E * I_y / L ** 2
        k_local[4, 2] = -6 * E * I_y / L ** 2
        k_local[4, 4] = 4 * E * I_y / L
        k_local[4, 8] = 6 * E * I_y / L ** 2
        k_local[4, 10] = 2 * E * I_y / L
        k_local[8, 2] = -12 * E * I_y / L ** 3
        k_local[8, 4] = 6 * E * I_y / L ** 2
        k_local[8, 8] = 12 * E * I_y / L ** 3
        k_local[8, 10] = 6 * E * I_y / L ** 2
        k_local[10, 2] = -6 * E * I_y / L ** 2
        k_local[10, 4] = 2 * E * I_y / L
        k_local[10, 8] = 6 * E * I_y / L ** 2
        k_local[10, 10] = 4 * E * I_y / L
        k_local[3, 3] = G * J / L
        k_local[3, 9] = -G * J / L
        k_local[9, 3] = -G * J / L
        k_local[9, 9] = G * J / L
        K_elem = T.T @ k_local @ T
        dof_i = [6 * node_i, 6 * node_i + 1, 6 * node_i + 2, 6 * node_i + 3, 6 * node_i + 4, 6 * node_i + 5]
        dof_j = [6 * node_j, 6 * node_j + 1, 6 * node_j + 2, 6 * node_j + 3, 6 * node_j + 4, 6 * node_j + 5]
        all_dofs = dof_i + dof_j
        for (i, di) in enumerate(all_dofs):
            for (j, dj) in enumerate(all_dofs):
                K[di, dj] += K_elem[i, j]
    for (node_idx, loads) in nodal_loads.items():
        if len(loads) != 6:
            raise ValueError('Each nodal load must have exactly 6 components: [Fx, Fy, Fz, Mx, My, Mz]')
        for i in range(6):
            P[6 * node_idx + i] = loads[i]
    fixed_dofs = []
    for (node_idx, bc) in boundary_conditions.items():
        if len(bc) != 6:
            raise ValueError('Each boundary condition must have exactly 6 values (0 or 1)')
        for i in range(6):
            if bc[i] == 1:
                fixed_dofs.append(6 * node_idx + i)
    free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
    if len(fixed_dofs) == n_dof:
        raise ValueError('All DOFs are constrained; no free DOFs remain')
    K_free = K[np.ix_(free_dofs, free_dofs)]
    P_free = P[free_dofs]
    try:
        u_free = np.linalg.solve(K_free, P_free)
    except np.linalg.LinAlgError:
        raise ValueError('Stiffness matrix is singular or ill-conditioned')
    u = np.zeros(n_dof)
    u[free_dofs] = u_free
    K_g = np.zeros((n_dof, n_dof))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        local_z = elem.get('local_z')
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        dx = (xj - xi) / L
        dy = (yj - yi) / L
        dz = (zj - zi) / L
        if local_z is None:
            if abs(dz) > 0.999:
                local_z = np.array([0.0, 1.0, 0.0])
            else:
                local_z = np.array([0.0, 0.0, 1.0])
        local_z = local_z / np.linalg.norm(local_z)
        local_x = np.array([dx, dy, dz])
        local_y = np.cross(local_z, local_x)
        local_y = local_y / np.linalg.norm(local_y)
        local_z = np.cross(local_x, local_y)
        T = np.zeros((12, 12))
        for i in range(3):
            T[i, i] = local_x[i]
            T[i, i + 3] = local_y[i]
            T[i, i + 6] = local_z[i]
            T[i + 3, i] = local_x[i]
            T[i + 3, i + 3] = local_y[i]
            T[i + 3, i + 6] = local_z[i]
            T[i + 6, i] = local_x[i]
            T[i + 6, i + 3] = local_y[i]
            T[i + 6, i + 6] = local_z[i]
            T[i + 9, i] = local_x[i]
            T[i + 9, i + 3] = local_y[i]
            T[i + 9, i + 6] = local_z[i]
        u_elem = np.zeros(12)
        u_elem[0:6] = u[6 * node_i:6 * node_i + 6]
        u_elem[6:12] = u[6 * node_j:6 * node_j + 6]
        u_local = T @ u_elem
        F_axial = E * A / L * (u_local[6] - u_local[0])
        k_g_local = np.zeros((12, 12))
        k_g_local[0, 0] = F_axial / L
        k_g_local[0, 6] = -F_axial / L
        k_g_local[6, 0] = -F_axial / L
        k_g_local[6, 6] = F_axial / L
        k_g_local[1, 1] = 6 * F_axial / (5 * L)
        k_g_local[1, 7] = -6 * F_axial / (5 * L)
        k_g_local[7, 1] = -6 * F_axial / (5 * L)
        k_g_local[7, 7] = 6 * F_axial / (5 * L)
        k_g_local[2, 2] = 6 * F_axial / (5 * L)
        k_g_local[2, 8] = -6 * F_axial / (5 * L)
        k_g_local[8, 2] = -6 * F_axial / (5 * L)
        k_g_local[8, 8] = 6 * F_axial / (5 * L)
        k_g_local[1, 7] = -6 * F_axial / (5 * L)
        k_g_local[7, 1] = -6 * F_axial / (5 * L)
        k_g_local[1, 5] = F_axial / 10
        k_g_local[1, 11] = F_axial / 10
        k_g_local[5, 1] = F_axial / 10
        k_g_local[11, 1] = F_axial / 10
        k_g_local[7, 5] = -F_axial / 10
        k_g_local[7, 11] = -F_axial / 10
        k_g_local[5, 7] = -F_axial / 10
        k_g_local[11, 7] = -F_axial / 10
        k_g_local[2, 4] = -F_axial / 10
        k_g_local[2, 10] = -F_axial / 10
        k_g_local[4, 2] = -F_axial / 10
        k_g_local[10, 2] = -F_axial / 10
        k_g_local[8, 4] = F_axial / 10
        k_g_local[8, 10] = F_axial / 10
        k_g_local[4, 8] = F_axial / 10
        k_g_local[10, 8] = F_axial / 10
        k_g_local[3, 3] = F_axial / 10
        k_g_local[3, 9] = -F_axial / 10
        k_g_local[9, 3] = -F_axial / 10
        k_g_local[9, 9] = F_axial / 10
        K_g_elem = T.T @ k_g_local @ T
        dof_i = [6 * node_i, 6 * node_i + 1, 6 * node_i + 2, 6 * node_i + 3, 6 * node_i + 4, 6 * node_i + 5]
        dof_j = [6 * node_j, 6 * node_j + 1, 6 * node_j + 2, 6 * node_j + 3, 6 * node_j + 4, 6 * node_j + 5]
        all_dofs = dof_i + dof_j
        for (i, di) in enumerate(all_dofs):
            for (j, dj) in enumerate(all_dofs):
                K_g[di, dj] += K_g_elem[i, j]
    K_free_free = K[np.ix_(free_dofs, free_dofs)]
    K_g_free_free = K_g[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvals, eigenvecs) = scipy.linalg.eig(K_g_free_free, K_free_free)
    except:
        raise ValueError('Eigenvalue problem failed to solve')
    real_eigenvals = []
    real_eigenvecs = []
    for i in range(len(eigenvals)):
        if np.isreal(eigenvals[i]) and abs(np.imag(eigenvals[i])) < 1e-10:
            real_val = np.real(eigenvals[i])
            if real_val < 0:
                lambda_val = -1.0 / real_val
                if lambda_val > 0:
                    real_eigenvals.append(lambda_val)
                    real_eigenvecs.append(eigenvecs[:, i])
    if len(real_eigenvals) == 0:
        raise ValueError('No positive eigenvalue found')
    min_idx = np.argmin(real_eigenvals)
    elastic_critical_load_factor = real_eigenvals[min_idx]
    mode_free = real_eigenvecs[min_idx]
    deformed_shape_vector = np.zeros(n_dof)
    deformed_shape_vector[free_dofs] = mode_free
    return (elastic_critical_load_factor, deformed_shape_vector)