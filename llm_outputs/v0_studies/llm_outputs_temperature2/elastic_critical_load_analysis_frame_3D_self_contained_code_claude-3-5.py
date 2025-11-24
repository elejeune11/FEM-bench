def elastic_critical_load_analysis_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = len(node_coords)
    n_dofs = 6 * n_nodes
    bc_flags = np.zeros(n_dofs, dtype=bool)
    for (node_idx, dof_spec) in boundary_conditions.items():
        base_idx = 6 * node_idx
        if all((isinstance(x, bool) for x in dof_spec)):
            bc_flags[base_idx:base_idx + 6] = dof_spec
        else:
            bc_flags[base_idx + np.array(dof_spec)] = True
    free_dofs = np.where(~bc_flags)[0]
    K = np.zeros((n_dofs, n_dofs))
    Kg = np.zeros((n_dofs, n_dofs))
    P = np.zeros(n_dofs)
    for (node_idx, loads) in nodal_loads.items():
        P[6 * node_idx:6 * node_idx + 6] = loads
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        xi = node_coords[node_i]
        xj = node_coords[node_j]
        dx = xj - xi
        L = np.linalg.norm(dx)
        ex = dx / L
        if elem['local_z'] is not None:
            ez_temp = np.array(elem['local_z'])
            ez = ez_temp - np.dot(ez_temp, ex) * ex
            ez = ez / np.linalg.norm(ez)
        else:
            if abs(ex[2]) < 0.99:
                ez_temp = np.array([0, 0, 1])
            else:
                ez_temp = np.array([1, 0, 0])
            ez = np.cross(ex, ez_temp)
            ez = ez / np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        T = np.zeros((3, 3))
        T[:, 0] = ex
        T[:, 1] = ey
        T[:, 2] = ez
        Lambda = np.zeros((12, 12))
        for i in range(4):
            Lambda[3 * i:3 * i + 3, 3 * i:3 * i + 3] = T
        EA_L = elem['E'] * elem['A'] / L
        EIy_L3 = elem['E'] * elem['Iy'] / L ** 3
        EIz_L3 = elem['E'] * elem['Iz'] / L ** 3
        GJ_L = elem['E'] / (2 * (1 + elem['nu'])) * elem['J'] / L
        ke = np.zeros((12, 12))
        ke[0, 0] = ke[6, 6] = EA_L
        ke[0, 6] = ke[6, 0] = -EA_L
        ke[3, 3] = ke[9, 9] = GJ_L
        ke[3, 9] = ke[9, 3] = -GJ_L
        ke[2, 2] = ke[8, 8] = 12 * EIy_L3
        ke[2, 8] = ke[8, 2] = -12 * EIy_L3
        ke[2, 4] = ke[4, 2] = 6 * L * EIy_L3
        ke[2, 10] = ke[10, 2] = -6 * L * EIy_L3
        ke[4, 4] = ke[10, 10] = 4 * L * L * EIy_L3
        ke[4, 8] = ke[8, 4] = -6 * L * EIy_L3
        ke[4, 10] = ke[10, 4] = 2 * L * L * EIy_L3
        ke[8, 10] = ke[10, 8] = 6 * L * EIy_L3
        ke[1, 1] = ke[7, 7] = 12 * EIz_L3
        ke[1, 7] = ke[7, 1] = -12 * EIz_L3
        ke[1, 5] = ke[5, 1] = -6 * L * EIz_L3
        ke[1, 11] = ke[11, 1] = 6 * L * EIz_L3
        ke[5, 5] = ke[11, 11] = 4 * L * L * EIz_L3
        ke[5, 7] = ke[7, 5] = 6 * L * EIz_L3
        ke[5, 11] = ke[11, 5] = 2 * L * L * EIz_L3
        ke[7, 11] = ke[11, 7] = -6 * L * EIz_L3
        ke_global = Lambda.T @ ke @ Lambda
        P_local = Lambda @ P[np.r_[6 * node_i:6 * node_i + 6, 6 * node_j:6 * node_j + 6]]
        N = -P_local[0]
        kg = np.zeros((12, 12))
        kg[1, 1] = kg[7, 7] = 6 / 5
        kg[1, 7] = kg[7, 1] = -6 / 5
        kg[2, 2] = kg[8, 8] = 6 / 5
        kg[2, 8] = kg[8, 2] = -6 / 5
        kg[4, 4] = kg[10, 10] = 2 * L * L / 15
        kg[4, 10] = kg[10, 4] = -L * L / 30
        kg[5, 5] = kg[11, 11] = 2 * L * L / 15
        kg[5, 11] = kg[11, 5] = -L * L / 30
        kg[1, 5] = kg[5, 1] = -L / 10
        kg[1, 11] = kg[11, 1] = -L / 10
        kg[2, 4] = kg[4, 2] = L / 10
        kg[2, 10] = kg[10, 2] = L / 10
        kg[4, 8] = kg[8, 4] = -L / 10
        kg[5, 7] = kg[7, 5] = L / 10
        kg[7, 11] = kg[11, 7] = L / 10
        kg[8, 10] = kg[10, 8] = -L / 10
        kg_global = N / L * (Lambda.T @ kg @ Lambda)
        dofs = np.r_[6 * node_i:6 * node_i + 6, 6 * node_j:6 * node_j + 6]
        K[np.ix_(dofs, dofs)] += ke_global
        Kg[np.ix_(dofs, dofs)] += kg_global
    u = np.zeros(n_dofs)
    u[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], P[free_dofs])
    K_red = K[np.ix_(free_dofs, free_dofs)]
    Kg_red = Kg[np.ix_(free_dofs, free_dofs)]
    (eigvals, eigvecs) = scipy.linalg.eigh(K_red, -Kg_red)
    pos_eigvals = eigvals[eigvals > 0]
    if len(pos_eigvals) == 0:
        raise ValueError('No positive eigenvalues found')
    lambda_cr = np.min(pos_eigvals)
    mode_idx = np.where(eigvals == lambda_cr)[0][0]
    mode_shape = np.zeros(n_dofs)
    mode_shape[free_dofs] = eigvecs[:, mode_idx]
    return (lambda_cr, mode_shape)