def elastic_critical_load_analysis_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    Kg = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    constrained_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        node_dofs = slice(6 * node_idx, 6 * (node_idx + 1))
        if isinstance(bc_spec[0], bool):
            fixed_local = [i for (i, fixed) in enumerate(bc_spec) if fixed]
        else:
            fixed_local = bc_spec
        constrained_dofs.update((6 * node_idx + i for i in fixed_local))
    free_dofs = list(set(range(n_dof)) - constrained_dofs)
    free_dofs.sort()
    for (node_idx, loads) in nodal_loads.items():
        P[6 * node_idx:6 * (node_idx + 1)] = loads
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
            if abs(ex[2]) < 0.9:
                ez_temp = np.array([0, 0, 1])
            else:
                ez_temp = np.array([0, 1, 0])
            ez = ez_temp - np.dot(ez_temp, ex) * ex
            ez = ez / np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        T = np.zeros((12, 12))
        R = np.column_stack([ex, ey, ez])
        for i in range(4):
            T[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = R
        (E, A) = (elem['E'], elem['A'])
        (Iy, Iz) = (elem['Iy'], elem['Iz'])
        J = elem['J']
        nu = elem['nu']
        G = E / (2 * (1 + nu))
        ke = np.zeros((12, 12))
        ke[0, 0] = ke[6, 6] = E * A / L
        ke[0, 6] = ke[6, 0] = -E * A / L
        ke[3, 3] = ke[9, 9] = G * J / L
        ke[3, 9] = ke[9, 3] = -G * J / L
        EIz = E * Iz
        ke[1, 1] = ke[7, 7] = 12 * EIz / L ** 3
        ke[1, 7] = ke[7, 1] = -12 * EIz / L ** 3
        ke[1, 5] = ke[5, 1] = 6 * EIz / L ** 2
        ke[1, 11] = ke[11, 1] = 6 * EIz / L ** 2
        ke[5, 5] = ke[11, 11] = 4 * EIz / L
        ke[5, 7] = ke[7, 5] = -6 * EIz / L ** 2
        ke[5, 11] = ke[11, 5] = 2 * EIz / L
        ke[7, 11] = ke[11, 7] = -6 * EIz / L ** 2
        EIy = E * Iy
        ke[2, 2] = ke[8, 8] = 12 * EIy / L ** 3
        ke[2, 8] = ke[8, 2] = -12 * EIy / L ** 3
        ke[2, 4] = ke[4, 2] = -6 * EIy / L ** 2
        ke[2, 10] = ke[10, 2] = -6 * EIy / L ** 2
        ke[4, 4] = ke[10, 10] = 4 * EIy / L
        ke[4, 8] = ke[8, 4] = 6 * EIy / L ** 2
        ke[4, 10] = ke[10, 4] = 2 * EIy / L
        ke[8, 10] = ke[10, 8] = 6 * EIy / L ** 2
        ke_global = T.T @ ke @ T
        dofs_i = slice(6 * elem['node_i'], 6 * (elem['node_i'] + 1))
        dofs_j = slice(6 * elem['node_j'], 6 * (elem['node_j'] + 1))
        dofs = np.r_[dofs_i, dofs_j]
        K[np.ix_(dofs, dofs)] += ke_global
    Kff = K[np.ix_(free_dofs, free_dofs)]
    Pf = P[free_dofs]
    uf = scipy.linalg.solve(Kff, Pf)
    u = np.zeros(n_dof)
    u[free_dofs] = uf
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
            if abs(ex[2]) < 0.9:
                ez_temp = np.array([0, 0, 1])
            else:
                ez_temp = np.array([0, 1, 0])
            ez = ez_temp - np.dot(ez_temp, ex) * ex
            ez = ez / np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        T = np.zeros((12, 12))
        R = np.column_stack([ex, ey, ez])
        for k in range(4):
            T[3 * k:3 * (k + 1), 3 * k:3 * (k + 1)] = R
        ue = np.concatenate([u[6 * i:6 * (i + 1)], u[6 * j:6 * j + 6]])
        ue_local = T @ ue
        N = elem['E'] * elem['A'] * (ue_local[6] - ue_local[0]) / L
        kg = np.zeros((12, 12))
        kg[1, 1] = kg[7, 7] = 6 / 5
        kg[1, 7] = kg[7, 1] = -6 / 5
        kg[1, 5] = kg[5, 1] = L / 10
        kg[1, 11] = kg[11, 1] = L / 10
        kg[2, 2] = kg[8, 8] = 6 / 5
        kg[2, 8] = kg[8, 2] = -6 / 5
        kg[2, 4] = kg[4, 2] = -L / 10
        kg[2, 10] = kg[10, 2] = -L / 10
        kg[4, 4] = kg[10, 10] = 2 * L ** 2 / 15
        kg[4, 8] = kg[8, 4] = -L / 10
        kg[4, 10] = kg[10, 4] = -L ** 2 / 30
        kg[5, 5] = kg[11, 11] = 2 * L ** 2 / 15
        kg[5, 7] = kg[7, 5] = -L / 10
        kg[5, 11] = kg[11, 5] = -L ** 2 / 30
        kg[8, 10] = kg[10, 8] = -L / 10
        kg[7, 11] = kg[11, 7] = -L / 10
        kg *= N / L
        kg_global = T.T @ kg @ T
        dofs_i = slice(6 * elem['node_i'], 6 * (elem['node_i'] + 1))
        dofs_j = slice(6 * elem['node_j'], 6 * (elem['node_j'] + 1))
        dofs = np.r_[dofs_i, dofs_j]
        Kg[np.ix_(dofs, dofs)] += kg_global
    Kff = K[np.ix_(free_dofs, free_dofs)]
    Kgff = Kg[np.ix_(free_dofs, free_dofs)]
    (eigenvals, eigenvecs) = scipy.linalg.eigh(Kff, -Kgff)
    pos_eigenvals = eigenvals[eigenvals > 0]
    if len(pos_eigenvals) == 0:
        raise ValueError('No positive eigenvalues found')
    critical_lambda = np.min(pos_eigenvals)
    critical_mode_idx = np.where(eigenvals == critical_lambda)[0][0]
    mode_shape = np.zeros(n_dof)
    mode_shape[free_dofs] = eigenvecs[:, critical_mode_idx]
    return (critical_lambda, mode_shape)