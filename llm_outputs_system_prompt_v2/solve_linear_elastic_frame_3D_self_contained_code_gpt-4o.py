def solve_linear_elastic_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    num_nodes = node_coords.shape[0]
    dof_per_node = 6
    total_dofs = num_nodes * dof_per_node
    K_global = np.zeros((total_dofs, total_dofs))
    P_global = np.zeros(total_dofs)
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        E = element['E']
        nu = element['nu']
        A = element['A']
        I_y = element['I_y']
        I_z = element['I_z']
        J = element['J']
        local_z = element['local_z']
        coords_i = node_coords[node_i]
        coords_j = node_coords[node_j]
        L = np.linalg.norm(coords_j - coords_i)
        G = E / (2 * (1 + nu))
        k_local = np.zeros((12, 12))
        k_local[0, 0] = k_local[6, 6] = E * A / L
        k_local[0, 6] = k_local[6, 0] = -E * A / L
        k_local[1, 1] = k_local[7, 7] = 12 * E * I_z / L ** 3
        k_local[1, 7] = k_local[7, 1] = -12 * E * I_z / L ** 3
        k_local[1, 5] = k_local[5, 1] = 6 * E * I_z / L ** 2
        k_local[1, 11] = k_local[11, 1] = -6 * E * I_z / L ** 2
        k_local[5, 7] = k_local[7, 5] = -6 * E * I_z / L ** 2
        k_local[5, 11] = k_local[11, 5] = 2 * E * I_z / L
        k_local[7, 11] = k_local[11, 7] = 6 * E * I_z / L ** 2
        k_local[11, 11] = 4 * E * I_z / L
        k_local[2, 2] = k_local[8, 8] = 12 * E * I_y / L ** 3
        k_local[2, 8] = k_local[8, 2] = -12 * E * I_y / L ** 3
        k_local[2, 4] = k_local[4, 2] = -6 * E * I_y / L ** 2
        k_local[2, 10] = k_local[10, 2] = 6 * E * I_y / L ** 2
        k_local[4, 8] = k_local[8, 4] = 6 * E * I_y / L ** 2
        k_local[4, 10] = k_local[10, 4] = 2 * E * I_y / L
        k_local[8, 10] = k_local[10, 8] = -6 * E * I_y / L ** 2
        k_local[10, 10] = 4 * E * I_y / L
        k_local[3, 3] = k_local[9, 9] = G * J / L
        k_local[3, 9] = k_local[9, 3] = -G * J / L
        T = np.eye(12)
        if local_z is not None:
            local_x = (coords_j - coords_i) / L
            local_y = np.cross(local_z, local_x)
            local_y /= np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
            R = np.zeros((3, 3))
            R[:, 0] = local_x
            R[:, 1] = local_y
            R[:, 2] = local_z
            T[:3, :3] = R
            T[3:6, 3:6] = R
            T[6:9, 6:9] = R
            T[9:, 9:] = R
        k_global = T.T @ k_local @ T
        dof_map = np.array([node_i * dof_per_node, node_i * dof_per_node + 1, node_i * dof_per_node + 2, node_i * dof_per_node + 3, node_i * dof_per_node + 4, node_i * dof_per_node + 5, node_j * dof_per_node, node_j * dof_per_node + 1, node_j * dof_per_node + 2, node_j * dof_per_node + 3, node_j * dof_per_node + 4, node_j * dof_per_node + 5])
        for i in range(12):
            for j in range(12):
                K_global[dof_map[i], dof_map[j]] += k_global[i, j]
    for (node, loads) in nodal_loads.items():
        for i in range(dof_per_node):
            P_global[node * dof_per_node + i] += loads[i]
    free_dofs = []
    fixed_dofs = []
    for node in range(num_nodes):
        if node in boundary_conditions:
            for (i, bc) in enumerate(boundary_conditions[node]):
                if bc == 0:
                    free_dofs.append(node * dof_per_node + i)
                else:
                    fixed_dofs.append(node * dof_per_node + i)
        else:
            for i in range(dof_per_node):
                free_dofs.append(node * dof_per_node + i)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_fc = K_global[np.ix_(free_dofs, fixed_dofs)]
    P_f = P_global[free_dofs]
    if np.linalg.cond(K_ff) >= 1e+16:
        raise ValueError('The free-free submatrix K_ff is ill-conditioned.')
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(total_dofs)
    u[free_dofs] = u_f
    r = np.zeros(total_dofs)
    r[fixed_dofs] = K_global[np.ix_(fixed_dofs, free_dofs)] @ u_f + K_global[np.ix_(fixed_dofs, fixed_dofs)] @ u[fixed_dofs] - P_global[fixed_dofs]
    return (u, r)