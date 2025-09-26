def solve_linear_elastic_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K_global = np.zeros((n_dofs, n_dofs))
    P_global = np.zeros(n_dofs)
    for element in elements:
        i = element['node_i']
        j = element['node_j']
        E = element['E']
        nu = element['nu']
        A = element['A']
        I_y = element['I_y']
        I_z = element['I_z']
        J = element['J']
        local_z = element.get('local_z')
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        dx = xj - xi
        dy = yj - yi
        dz = zj - zi
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if local_z is None:
            if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                local_z = np.array([0.0, 1.0, 0.0])
            else:
                local_z = np.array([-dy, dx, 0.0])
                norm = np.linalg.norm(local_z)
                if norm < 1e-12:
                    local_z = np.array([0.0, 1.0, 0.0])
                else:
                    local_z /= norm
        local_x = np.array([dx, dy, dz]) / L
        local_y = np.cross(local_z, local_x)
        local_y /= np.linalg.norm(local_y)
        local_z = np.cross(local_x, local_y)
        T = np.zeros((12, 12))
        R = np.column_stack([local_x, local_y, local_z])
        for k in range(4):
            T[3 * k:3 * k + 3, 3 * k:3 * k + 3] = R
        G = E / (2 * (1 + nu))
        phi_y = 12 * E * I_z / (G * A * L ** 2) if G * A > 1e-12 else 0.0
        phi_z = 12 * E * I_y / (G * A * L ** 2) if G * A > 1e-12 else 0.0
        k_local = np.zeros((12, 12))
        k_local[0, 0] = E * A / L
        k_local[0, 6] = -E * A / L
        k_local[6, 0] = -E * A / L
        k_local[6, 6] = E * A / L
        k_local[1, 1] = 12 * E * I_z / (L ** 3 * (1 + phi_y))
        k_local[1, 5] = 6 * E * I_z / (L ** 2 * (1 + phi_y))
        k_local[1, 7] = -12 * E * I_z / (L ** 3 * (1 + phi_y))
        k_local[1, 11] = 6 * E * I_z / (L ** 2 * (1 + phi_y))
        k_local[5, 1] = 6 * E * I_z / (L ** 2 * (1 + phi_y))
        k_local[5, 5] = (4 + phi_y) * E * I_z / (L * (1 + phi_y))
        k_local[5, 7] = -6 * E * I_z / (L ** 2 * (1 + phi_y))
        k_local[5, 11] = (2 - phi_y) * E * I_z / (L * (1 + phi_y))
        k_local[7, 1] = -12 * E * I_z / (L ** 3 * (1 + phi_y))
        k_local[7, 5] = -6 * E * I_z / (L ** 2 * (1 + phi_y))
        k_local[7, 7] = 12 * E * I_z / (L ** 3 * (1 + phi_y))
        k_local[7, 11] = -6 * E * I_z / (L ** 2 * (1 + phi_y))
        k_local[11, 1] = 6 * E * I_z / (L ** 2 * (1 + phi_y))
        k_local[11, 5] = (2 - phi_y) * E * I_z / (L * (1 + phi_y))
        k_local[11, 7] = -6 * E * I_z / (L ** 2 * (1 + phi_y))
        k_local[11, 11] = (4 + phi_y) * E * I_z / (L * (1 + phi_y))
        k_local[2, 2] = 12 * E * I_y / (L ** 3 * (1 + phi_z))
        k_local[2, 4] = -6 * E * I_y / (L ** 2 * (1 + phi_z))
        k_local[2, 8] = -12 * E * I_y / (L ** 3 * (1 + phi_z))
        k_local[2, 10] = -6 * E * I_y / (L ** 2 * (1 + phi_z))
        k_local[4, 2] = -6 * E * I_y / (L ** 2 * (1 + phi_z))
        k_local[4, 4] = (4 + phi_z) * E * I_y / (L * (1 + phi_z))
        k_local[4, 8] = 6 * E * I_y / (L ** 2 * (1 + phi_z))
        k_local[4, 10] = (2 - phi_z) * E * I_y / (L * (1 + phi_z))
        k_local[8, 2] = -12 * E * I_y / (L ** 3 * (1 + phi_z))
        k_local[8, 4] = 6 * E * I_y / (L ** 2 * (1 + phi_z))
        k_local[8, 8] = 12 * E * I_y / (L ** 3 * (1 + phi_z))
        k_local[8, 10] = 6 * E * I_y / (L ** 2 * (1 + phi_z))
        k_local[10, 2] = -6 * E * I_y / (L ** 2 * (1 + phi_z))
        k_local[10, 4] = (2 - phi_z) * E * I_y / (L * (1 + phi_z))
        k_local[10, 8] = 6 * E * I_y / (L ** 2 * (1 + phi_z))
        k_local[10, 10] = (4 + phi_z) * E * I_y / (L * (1 + phi_z))
        k_local[3, 3] = G * J / L
        k_local[3, 9] = -G * J / L
        k_local[9, 3] = -G * J / L
        k_local[9, 9] = G * J / L
        k_global_elem = T.T @ k_local @ T
        dofs_i = [6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5]
        dofs_j = [6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5]
        all_dofs = dofs_i + dofs_j
        for (idx1, dof1) in enumerate(all_dofs):
            for (idx2, dof2) in enumerate(all_dofs):
                K_global[dof1, dof2] += k_global_elem[idx1, idx2]
    for (node, load) in nodal_loads.items():
        start_dof = 6 * node
        P_global[start_dof:start_dof + 6] = load
    fixed_dofs = []
    for node in range(n_nodes):
        if node in boundary_conditions:
            bc = boundary_conditions[node]
            for dof_local in range(6):
                if bc[dof_local] == 1:
                    fixed_dofs.append(6 * node + dof_local)
    free_dofs = [dof for dof in range(n_dofs) if dof not in fixed_dofs]
    if free_dofs:
        K_ff = K_global[np.ix_(free_dofs, free_dofs)]
        cond_num = np.linalg.cond(K_ff)
        if cond_num >= 1e+16:
            raise ValueError('Free-free submatrix is ill-conditioned')
        P_f = P_global[free_dofs]
        u_f = np.linalg.solve(K_ff, P_f)
        u = np.zeros(n_dofs)
        u[free_dofs] = u_f
    else:
        u = np.zeros(n_dofs)
    r = K_global @ u - P_global
    return (u, r)