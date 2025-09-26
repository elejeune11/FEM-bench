def solve_linear_elastic_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K_global = np.zeros((n_dofs, n_dofs))
    P_global = np.zeros(n_dofs)
    for element in elements:
        (i, j) = (element['node_i'], element['node_j'])
        (E, nu) = (element['E'], element['nu'])
        (A, I_y, I_z, J) = (element['A'], element['I_y'], element['I_z'], element['J'])
        local_z = element.get('local_z')
        coords_i = node_coords[i]
        coords_j = node_coords[j]
        L_vec = coords_j - coords_i
        L = np.linalg.norm(L_vec)
        if L < 1e-12:
            continue
        e_x = L_vec / L
        if local_z is None:
            if abs(e_x[2]) > 0.9:
                local_z = np.array([0.0, 1.0, 0.0])
            else:
                local_z = np.array([0.0, 0.0, 1.0])
        e_z = local_z - np.dot(local_z, e_x) * e_x
        e_z_norm = np.linalg.norm(e_z)
        if e_z_norm < 1e-12:
            if abs(e_x[2]) > 0.9:
                e_z = np.array([0.0, 1.0, 0.0])
            else:
                e_z = np.array([0.0, 0.0, 1.0])
            e_z = e_z - np.dot(e_z, e_x) * e_x
        e_z = e_z / np.linalg.norm(e_z)
        e_y = np.cross(e_z, e_x)
        e_y = e_y / np.linalg.norm(e_y)
        e_z = np.cross(e_x, e_y)
        R = np.column_stack((e_x, e_y, e_z))
        T = np.zeros((12, 12))
        for block in range(4):
            T[3 * block:3 * block + 3, 3 * block:3 * block + 3] = R
        G = E / (2 * (1 + nu))
        k_local = np.zeros((12, 12))
        k_local[0, 0] = E * A / L
        k_local[0, 6] = -E * A / L
        k_local[6, 0] = -E * A / L
        k_local[6, 6] = E * A / L
        k_local[3, 3] = G * J / L
        k_local[3, 9] = -G * J / L
        k_local[9, 3] = -G * J / L
        k_local[9, 9] = G * J / L
        phi_y = 12 * E * I_z / L ** 3
        k_local[1, 1] = phi_y
        k_local[1, 5] = 6 * E * I_z / L ** 2
        k_local[1, 7] = -phi_y
        k_local[1, 11] = 6 * E * I_z / L ** 2
        k_local[5, 1] = 6 * E * I_z / L ** 2
        k_local[5, 5] = 4 * E * I_z / L
        k_local[5, 7] = -6 * E * I_z / L ** 2
        k_local[5, 11] = 2 * E * I_z / L
        k_local[7, 1] = -phi_y
        k_local[7, 5] = -6 * E * I_z / L ** 2
        k_local[7, 7] = phi_y
        k_local[7, 11] = -6 * E * I_z / L ** 2
        k_local[11, 1] = 6 * E * I_z / L ** 2
        k_local[11, 5] = 2 * E * I_z / L
        k_local[11, 7] = -6 * E * I_z / L ** 2
        k_local[11, 11] = 4 * E * I_z / L
        phi_z = 12 * E * I_y / L ** 3
        k_local[2, 2] = phi_z
        k_local[2, 4] = -6 * E * I_y / L ** 2
        k_local[2, 8] = -phi_z
        k_local[2, 10] = -6 * E * I_y / L ** 2
        k_local[4, 2] = -6 * E * I_y / L ** 2
        k_local[4, 4] = 4 * E * I_y / L
        k_local[4, 8] = 6 * E * I_y / L ** 2
        k_local[4, 10] = 2 * E * I_y / L
        k_local[8, 2] = -phi_z
        k_local[8, 4] = 6 * E * I_y / L ** 2
        k_local[8, 8] = phi_z
        k_local[8, 10] = 6 * E * I_y / L ** 2
        k_local[10, 2] = -6 * E * I_y / L ** 2
        k_local[10, 4] = 2 * E * I_y / L
        k_local[10, 8] = 6 * E * I_y / L ** 2
        k_local[10, 10] = 4 * E * I_y / L
        k_global_elem = T.T @ k_local @ T
        dofs_i = range(6 * i, 6 * i + 6)
        dofs_j = range(6 * j, 6 * j + 6)
        dofs_elem = list(dofs_i) + list(dofs_j)
        for (idx1, dof1) in enumerate(dofs_elem):
            for (idx2, dof2) in enumerate(dofs_elem):
                K_global[dof1, dof2] += k_global_elem[idx1, idx2]
    for (node, loads) in nodal_loads.items():
        dof_start = 6 * node
        P_global[dof_start:dof_start + 6] = loads
    fixed_dofs = []
    for node in range(n_nodes):
        if node in boundary_conditions:
            bc = boundary_conditions[node]
            for (dof_local, fixed) in enumerate(bc):
                if fixed:
                    fixed_dofs.append(6 * node + dof_local)
    free_dofs = [dof for dof in range(n_dofs) if dof not in fixed_dofs]
    if free_dofs:
        K_ff = K_global[np.ix_(free_dofs, free_dofs)]
        cond_num = np.linalg.cond(K_ff)
        if cond_num >= 1e+16:
            raise ValueError(f'Ill-conditioned system: cond(K_ff) = {cond_num:.2e} >= 1e16')
        P_f = P_global[free_dofs]
        u_f = np.linalg.solve(K_ff, P_f)
        u = np.zeros(n_dofs)
        u[free_dofs] = u_f
        if fixed_dofs:
            K_rf = K_global[np.ix_(fixed_dofs, free_dofs)]
            P_r = P_global[fixed_dofs]
            r_fixed = K_rf @ u_f - P_r
            r = np.zeros(n_dofs)
            r[fixed_dofs] = r_fixed
        else:
            r = np.zeros(n_dofs)
    else:
        u = np.zeros(n_dofs)
        r = -P_global
    return (u, r)