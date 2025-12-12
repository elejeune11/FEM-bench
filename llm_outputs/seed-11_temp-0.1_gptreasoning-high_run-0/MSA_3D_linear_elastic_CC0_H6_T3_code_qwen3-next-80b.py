def MSA_3D_linear_elastic_CC0_H6_T3(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    N = len(node_coords)
    K_global = np.zeros((6 * N, 6 * N))
    F_global = np.zeros(6 * N)
    for elem in elements:
        i = elem['node_i']
        j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        local_z = np.array(elem.get('local_z', [0, 0, 1]), dtype=float)
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        if L == 0:
            continue
        (dx, dy, dz) = ((xj - xi) / L, (yj - yi) / L, (zj - zi) / L)
        beam_axis = np.array([dx, dy, dz])
        if np.abs(np.dot(beam_axis, local_z)) > 0.999:
            if abs(dz) < 0.999:
                local_z = np.array([0, 0, 1])
            else:
                local_z = np.array([0, 1, 0])
        local_z = local_z / np.linalg.norm(local_z)
        local_y = np.cross(beam_axis, local_z)
        local_y = local_y / np.linalg.norm(local_y)
        local_z = np.cross(local_y, beam_axis)
        local_z = local_z / np.linalg.norm(local_z)
        T = np.zeros((6, 6))
        T[:3, :3] = np.array([beam_axis, local_y, local_z]).T
        T[3:, 3:] = np.array([beam_axis, local_y, local_z]).T
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
        G = E / (2 * (1 + nu))
        dof_indices = [i * 6, i * 6 + 1, i * 6 + 2, i * 6 + 3, i * 6 + 4, i * 6 + 5, j * 6, j * 6 + 1, j * 6 + 2, j * 6 + 3, j * 6 + 4, j * 6 + 5]
        k_global_local = T.T @ k_local @ T
        for a in range(12):
            for b in range(12):
                K_global[dof_indices[a], dof_indices[b]] += k_global_local[a, b]
    for (node, bc) in boundary_conditions.items():
        for dof in range(6):
            if bc[dof] == 1:
                K_global[dof + node * 6, :] = 0
                K_global[:, dof + node * 6] = 0
                K_global[dof + node * 6, dof + node * 6] = 1
    for (node, loads) in nodal_loads.items():
        for dof in range(6):
            F_global[dof + node * 6] += loads[dof]
    free_dofs = []
    fixed_dofs = []
    for i in range(6 * N):
        if np.allclose(K_global[i, i], 1):
            fixed_dofs.append(i)
        else:
            free_dofs.append(i)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_ff_cond = np.linalg.cond(K_ff)
    if K_ff_cond >= 1e+16:
        raise ValueError('Free-free stiffness matrix is ill-conditioned')
    F_free = F_global[free_dofs]
    u_free = np.linalg.solve(K_ff, F_free)
    u = np.zeros(6 * N)
    u[free_dofs] = u_free
    r = K_global @ u - F_global
    return (u, r)