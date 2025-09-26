def solve_linear_elastic_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = len(node_coords)
    n_dofs = 6 * n_nodes
    K = np.zeros((n_dofs, n_dofs))
    P = np.zeros(n_dofs)
    for (node_idx, loads) in nodal_loads.items():
        dof_start = 6 * node_idx
        P[dof_start:dof_start + 6] = loads
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        dx = node_coords[j] - node_coords[i]
        L = np.linalg.norm(dx)
        (E, A) = (elem['E'], elem['A'])
        (Iy, Iz) = (elem['I_y'], elem['I_z'])
        J = elem['J']
        nu = elem['nu']
        G = E / (2 * (1 + nu))
        ex = dx / L
        if 'local_z' in elem and elem['local_z'] is not None:
            ez = elem['local_z'] / np.linalg.norm(elem['local_z'])
            ey = np.cross(ez, ex)
            ez = np.cross(ex, ey)
        else:
            if abs(ex[2]) < 0.999:
                ey = np.cross([0, 0, 1], ex)
            else:
                ey = np.cross([1, 0, 0], ex)
            ey /= np.linalg.norm(ey)
            ez = np.cross(ex, ey)
        T = np.zeros((12, 12))
        R = np.column_stack((ex, ey, ez))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        EA_L = E * A / L
        EIy_L3 = E * Iy / L ** 3
        EIz_L3 = E * Iz / L ** 3
        GJ_L = G * J / L
        k = np.zeros((12, 12))
        k[0, 0] = k[6, 6] = EA_L
        k[0, 6] = k[6, 0] = -EA_L
        k[3, 3] = k[9, 9] = GJ_L
        k[3, 9] = k[9, 3] = -GJ_L
        k[2, 2] = k[8, 8] = 12 * EIz_L3
        k[2, 8] = k[8, 2] = -12 * EIz_L3
        k[2, 4] = k[4, 2] = 6 * EIz_L3 * L
        k[2, 10] = k[10, 2] = -6 * EIz_L3 * L
        k[4, 4] = k[10, 10] = 4 * EIz_L3 * L * L
        k[4, 8] = k[8, 4] = -6 * EIz_L3 * L
        k[4, 10] = k[10, 4] = 2 * EIz_L3 * L * L
        k[8, 10] = k[10, 8] = 6 * EIz_L3 * L
        k[1, 1] = k[7, 7] = 12 * EIy_L3
        k[1, 7] = k[7, 1] = -12 * EIy_L3
        k[1, 5] = k[5, 1] = -6 * EIy_L3 * L
        k[1, 11] = k[11, 1] = 6 * EIy_L3 * L
        k[5, 5] = k[11, 11] = 4 * EIy_L3 * L * L
        k[5, 7] = k[7, 5] = 6 * EIy_L3 * L
        k[5, 11] = k[11, 5] = 2 * EIy_L3 * L * L
        k[7, 11] = k[11, 7] = -6 * EIy_L3 * L
        k_global = T.T @ k @ T
        dofs_i = slice(6 * elem['node_i'], 6 * elem['node_i'] + 6)
        dofs_j = slice(6 * elem['node_j'], 6 * elem['node_j'] + 6)
        dofs = np.r_[dofs_i, dofs_j]
        K[np.ix_(dofs, dofs)] += k_global
    fixed_dofs = []
    for (node, fixity) in boundary_conditions.items():
        for (dof, is_fixed) in enumerate(fixity):
            if is_fixed:
                fixed_dofs.append(6 * node + dof)
    fixed_dofs = np.array(fixed_dofs)
    free_dofs = np.setdiff1d(np.arange(n_dofs), fixed_dofs)
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_fs = K[np.ix_(free_dofs, fixed_dofs)]
    P_f = P[free_dofs]
    if np.linalg.cond(K_ff) >= 1e+16:
        raise ValueError('Free-free stiffness matrix is ill-conditioned')
    u = np.zeros(n_dofs)
    u[free_dofs] = np.linalg.solve(K_ff, P_f)
    r = np.zeros(n_dofs)
    r[fixed_dofs] = K[np.ix_(fixed_dofs, slice(None))] @ u - P[fixed_dofs]
    return (u, r)