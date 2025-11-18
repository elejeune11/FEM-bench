def elastic_critical_load_analysis_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    bc_bool = {}
    for (node, bc) in boundary_conditions.items():
        if isinstance(bc[0], bool):
            bc_bool[node] = bc
        else:
            temp = [False] * 6
            for idx in bc:
                temp[idx] = True
            bc_bool[node] = temp
    constrained = np.zeros(n_dof, dtype=bool)
    for (node, bc) in bc_bool.items():
        constrained[6 * node:6 * node + 6] = bc
    free = ~constrained
    free_idx = np.where(free)[0]
    K = np.zeros((n_dof, n_dof))
    Kg = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    for (node, loads) in nodal_loads.items():
        P[6 * node:6 * node + 6] = loads
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        xi = node_coords[i]
        xj = node_coords[j]
        L = np.linalg.norm(xj - xi)
        ex = (xj - xi) / L
        if elem['local_z'] is not None:
            ez_temp = np.array(elem['local_z'])
            ez = ez_temp - np.dot(ez_temp, ex) * ex
            ez = ez / np.linalg.norm(ez)
        else:
            ez_temp = np.array([0, 0, 1])
            ez = ez_temp - np.dot(ez_temp, ex) * ex
            if np.allclose(ez, 0):
                ez = np.array([0, 1, 0])
            else:
                ez = ez / np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        T = np.vstack((ex, ey, ez))
        R = np.zeros((12, 12))
        for i in range(4):
            R[3 * i:3 * i + 3, 3 * i:3 * i + 3] = T
        (E, nu) = (elem['E'], elem['nu'])
        A = elem['A']
        (Iy, Iz) = (elem['Iy'], elem['Iz'])
        J = elem['J']
        ke = np.zeros((12, 12))
        EA_L = E * A / L
        ke[0, 0] = ke[6, 6] = EA_L
        ke[0, 6] = ke[6, 0] = -EA_L
        GJ_L = E * J / (2 * (1 + nu)) / L
        ke[3, 3] = ke[9, 9] = GJ_L
        ke[3, 9] = ke[9, 3] = -GJ_L
        EIz_L3 = E * Iz / L ** 3
        EIy_L3 = E * Iy / L ** 3
        ke[1, 1] = ke[7, 7] = 12 * EIz_L3
        ke[1, 7] = ke[7, 1] = -12 * EIz_L3
        ke[1, 5] = ke[5, 1] = 6 * EIz_L3 * L
        ke[1, 11] = ke[11, 1] = 6 * EIz_L3 * L
        ke[5, 5] = ke[11, 11] = 4 * EIz_L3 * L * L
        ke[5, 7] = ke[7, 5] = -6 * EIz_L3 * L
        ke[5, 11] = ke[11, 5] = 2 * EIz_L3 * L * L
        ke[7, 11] = ke[11, 7] = -6 * EIz_L3 * L
        ke[2, 2] = ke[8, 8] = 12 * EIy_L3
        ke[2, 8] = ke[8, 2] = -12 * EIy_L3
        ke[2, 4] = ke[4, 2] = -6 * EIy_L3 * L
        ke[2, 10] = ke[10, 2] = -6 * EIy_L3 * L
        ke[4, 4] = ke[10, 10] = 4 * EIy_L3 * L * L
        ke[4, 8] = ke[8, 4] = 6 * EIy_L3 * L
        ke[4, 10] = ke[10, 4] = 2 * EIy_L3 * L * L
        ke[8, 10] = ke[10, 8] = 6 * EIy_L3 * L
        ke_global = R.T @ ke @ R
        I_rho = elem['I_rho']
        dof_i = slice(6 * elem['node_i'], 6 * elem['node_i'] + 6)
        dof_j = slice(6 * elem['node_j'], 6 * elem['node_j'] + 6)
        dofs = np.r_[dof_i, dof_j]
        K[np.ix_(dofs, dofs)] += ke_global
    u = np.zeros(n_dof)
    u[free_idx] = np.linalg.solve(K[np.ix_(free_idx, free_idx)], P[free_idx])
    for elem in elements:
        dof_i = slice(6 * elem['node_i'], 6 * elem['node_i'] + 6)
        dof_j = slice(6 * elem['node_j'], 6 * elem['node_j'] + 6)
        ue = np.r_[u[dof_i], u[dof_j]]
        (i, j) = (elem['node_i'], elem['node_j'])
        xi = node_coords[i]
        xj = node_coords[j]
        L = np.linalg.norm(xj - xi)
        ex = (xj - xi) / L
        delta = np.dot(ue[6:9] - ue[0:3], ex)
        N = elem['E'] * elem['A'] * delta / L
        kg = np.zeros((12, 12))
        N_L = N / L
        for i in range(2):
            kg[1 + i, 1 + i] = kg[7 + i, 7 + i] = 6 / 5 * N_L
            kg[1 + i, 7 + i] = kg[7 + i, 1 + i] = -6 / 5 * N_L