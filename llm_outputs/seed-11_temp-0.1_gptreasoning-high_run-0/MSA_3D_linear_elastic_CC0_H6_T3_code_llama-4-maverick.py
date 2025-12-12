def MSA_3D_linear_elastic_CC0_H6_T3(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    N = len(node_coords)
    K_global = np.zeros((6 * N, 6 * N))
    F_global = np.zeros(6 * N)
    for element in elements:
        (node_i, node_j) = (element['node_i'], element['node_j'])
        (x_i, y_i, z_i) = node_coords[node_i]
        (x_j, y_j, z_j) = node_coords[node_j]
        L = np.sqrt((x_j - x_i) ** 2 + (y_j - y_i) ** 2 + (z_j - z_i) ** 2)
        (E, nu, A, I_y, I_z, J) = (element['E'], element['nu'], element['A'], element['I_y'], element['I_z'], element['J'])
        G = E / (2 * (1 + nu))
        if 'local_z' in element and element['local_z'] is not None:
            local_z = np.array(element['local_z'])
        else:
            beam_axis = np.array([x_j - x_i, y_j - y_i, z_j - z_i]) / L
            if np.allclose(beam_axis, np.array([0, 0, 1])):
                local_z = np.array([0, 1, 0])
            else:
                local_z = np.array([0, 0, 1])
        local_x = np.cross(beam_axis, local_z)
        local_x = local_x / np.linalg.norm(local_x)
        local_z = np.cross(beam_axis, local_x)
        local_z = local_z / np.linalg.norm(local_z)
        T = np.column_stack((local_x, beam_axis, local_z))
        K_local = np.zeros((12, 12))
        K_local[0:3, 0:3] = np.array([[E * A / L, 0, 0], [0, 12 * E * I_z / L ** 3, 0], [0, 0, 12 * E * I_y / L ** 3]])
        K_local[0:3, 3:6] = np.array([[0, 0, 0], [0, 0, 6 * E * I_z / L ** 2], [0, -6 * E * I_y / L ** 2, 0]])
        K_local[0:3, 6:9] = np.array([[-E * A / L, 0, 0], [0, -12 * E * I_z / L ** 3, 0], [0, 0, -12 * E * I_y / L ** 3]])
        K_local[0:3, 9:12] = np.array([[0, 0, 0], [0, 0, 6 * E * I_z / L ** 2], [0, -6 * E * I_y / L ** 2, 0]])
        K_local[3:6, 0:3] = K_local[0:3, 3:6].T
        K_local[3:6, 3:6] = np.array([[G * J / L, 0, 0], [0, 4 * E * I_y / L, 0], [0, 0, 4 * E * I_z / L]])
        K_local[3:6, 6:9] = np.array([[0, 0, 0], [0, 0, -6 * E * I_y / L ** 2], [0, 6 * E * I_z / L ** 2, 0]])
        K_local[3:6, 9:12] = np.array([[-G * J / L, 0, 0], [0, 2 * E * I_y / L, 0], [0, 0, 2 * E * I_z / L]])
        K_local[6:9, 0:3] = K_local[0:3, 6:9].T
        K_local[6:9, 3:6] = K_local[3:6, 6:9].T
        K_local[6:9, 6:9] = np.array([[E * A / L, 0, 0], [0, 12 * E * I_z / L ** 3, 0], [0, 0, 12 * E * I_y / L ** 3]])
        K_local[6:9, 9:12] = np.array([[0, 0, 0], [0, 0, -6 * E * I_z / L ** 2], [0, 6 * E * I_y / L ** 2, 0]])
        K_local[9:12, 0:3] = K_local[0:3, 9:12].T
        K_local[9:12, 3:6] = K_local[3:6, 9:12].T
        K_local[9:12, 6:9] = K_local[6:9, 9:12].T
        K_local[9:12, 9:12] = np.array([[G * J / L, 0, 0], [0, 4 * E * I_y / L, 0], [0, 0, 4 * E * I_z / L]])
        T_expanded = np.block([[T, np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))], [np.zeros((3, 3)), T, np.zeros((3, 3)), np.zeros((3, 3))], [np.zeros((3, 3)), np.zeros((3, 3)), T, np.zeros((3, 3))], [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), T]])
        K_global[np.ix_([6 * node_i, 6 * node_i + 1, 6 * node_i + 2, 6 * node_i + 3, 6 * node_i + 4, 6 * node_i + 5, 6 * node_j, 6 * node_j + 1, 6 * node_j + 2, 6 * node_j + 3, 6 * node_j + 4, 6 * node_j + 5], [6 * node_i, 6 * node_i + 1, 6 * node_i + 2, 6 * node_i + 3, 6 * node_i + 4, 6 * node_i + 5, 6 * node_j, 6 * node_j + 1, 6 * node_j + 2, 6 * node_j + 3, 6 * node_j + 4, 6 * node_j + 5])] += T_expanded @ K_local @ T_expanded.T
    for (node, loads) in nodal_loads.items():
        F_global[6 * node:6 * node + 6] += np.array(loads)
    fixed_dofs = set()
    for (node, bc) in boundary_conditions.items():
        for (i, fix) in enumerate(bc):
            if fix:
                fixed_dofs.add(6 * node + i)
    free_dofs = set(range(6 * N)) - fixed_dofs
    free_dofs = sorted(list(free_dofs))
    fixed_dofs = sorted(list(fixed_dofs))
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_fr = K_global[np.ix_(free_dofs, fixed_dofs)]
    K_rf = K_global[np.ix_(fixed_dofs, free_dofs)]
    K_rr = K_global[np.ix_(fixed_dofs, fixed_dofs)]
    F_f = F_global[free_dofs]
    F_r = F_global[fixed_dofs]
    if np.linalg.cond(K_ff) > 1e+16:
        raise ValueError('The free-free stiffness matrix is ill-conditioned.')
    u_f = np.linalg.solve(K_ff, F_f - K_fr @ np.zeros(len(fixed_dofs)))
    u = np.zeros(6 * N)
    u[free_dofs] = u_f
    r = K_rf @ u_f + K_rr @ np.zeros(len(fixed_dofs)) - F_r
    return (u, r)