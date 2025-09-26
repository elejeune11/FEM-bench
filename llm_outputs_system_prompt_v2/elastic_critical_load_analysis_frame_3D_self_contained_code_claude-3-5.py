def elastic_critical_load_analysis_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    Kg = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    for elem in elements:
        (node_i, node_j) = (elem['node_i'], elem['node_j'])
        xi = node_coords[node_i]
        xj = node_coords[node_j]
        L = np.linalg.norm(xj - xi)
        ex = (xj - xi) / L
        if elem['local_z'] is not None:
            ez_temp = np.array(elem['local_z'])
            ez = ez_temp - np.dot(ez_temp, ex) * ex
            ez = ez / np.linalg.norm(ez)
        else:
            ez_temp = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(ex, ez_temp)) > 0.99:
                ez_temp = np.array([0.0, 1.0, 0.0])
            ez = ez_temp - np.dot(ez_temp, ex) * ex
            ez = ez / np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        T = np.zeros((12, 12))
        R = np.column_stack((ex, ey, ez))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        ke = np.zeros((12, 12))
        (E, A) = (elem['E'], elem['A'])
        (Iy, Iz) = (elem['Iy'], elem['Iz'])
        (J, nu) = (elem['J'], elem['nu'])
        ke[0, 0] = ke[6, 6] = E * A / L
        ke[0, 6] = ke[6, 0] = -E * A / L
        G = E / (2 * (1 + nu))
        ke[3, 3] = ke[9, 9] = G * J / L
        ke[3, 9] = ke[9, 3] = -G * J / L
        ke[1, 1] = ke[7, 7] = 12 * E * Iz / L ** 3
        ke[1, 7] = ke[7, 1] = -12 * E * Iz / L ** 3
        ke[1, 5] = ke[5, 1] = 6 * E * Iz / L ** 2
        ke[1, 11] = ke[11, 1] = 6 * E * Iz / L ** 2
        ke[5, 5] = ke[11, 11] = 4 * E * Iz / L
        ke[5, 7] = ke[7, 5] = -6 * E * Iz / L ** 2
        ke[5, 11] = ke[11, 5] = 2 * E * Iz / L
        ke[7, 11] = ke[11, 7] = -6 * E * Iz / L ** 2
        ke[2, 2] = ke[8, 8] = 12 * E * Iy / L ** 3
        ke[2, 8] = ke[8, 2] = -12 * E * Iy / L ** 3
        ke[2, 4] = ke[4, 2] = -6 * E * Iy / L ** 2
        ke[2, 10] = ke[10, 2] = -6 * E * Iy / L ** 2
        ke[4, 4] = ke[10, 10] = 4 * E * Iy / L
        ke[4, 8] = ke[8, 4] = 6 * E * Iy / L ** 2
        ke[4, 10] = ke[10, 4] = 2 * E * Iy / L
        ke[8, 10] = ke[10, 8] = 6 * E * Iy / L ** 2
        ke_global = T.T @ ke @ T
        kg = np.zeros((12, 12))
        I_rho = elem['I_rho']
        dof_i = slice(6 * node_i, 6 * node_i + 6)
        dof_j = slice(6 * node_j, 6 * node_j + 6)
        dofs = np.r_[dof_i, dof_j]
        K[np.ix_(dofs, dofs)] += ke_global
        for (node, load) in nodal_loads.items():
            P[6 * node:6 * node + 6] = load
    free_dofs = np.ones(n_dof, dtype=bool)
    for (node, bc) in boundary_conditions.items():
        if isinstance(bc[0], bool):
            constrained = [i for (i, fixed) in enumerate(bc) if fixed]
        else:
            constrained = bc
        free_dofs[6 * node + np.array(constrained)] = False
    Kff = K[np.ix_(free_dofs, free_dofs)]
    Pf = P[free_dofs]
    uf = np.linalg.solve(Kff, Pf)
    u = np.zeros(n_dof)
    u[free_dofs] = uf
    for elem in elements:
        (node_i, node_j) = (elem['node_i'], elem['node_j'])
        xi = node_coords[node_i]
        xj = node_coords[node_j]
        L = np.linalg.norm(xj - xi)
        ex = (xj - xi) / L
        if elem['local_z'] is not None:
            ez_temp = np.array(elem['local_z'])
            ez = ez_temp - np.dot(ez_temp, ex) * ex
            ez = ez / np.linalg.norm(ez)
        else:
            ez_temp = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(ex, ez_temp)) > 0.99:
                ez_temp = np.array([0.0, 1.0, 0.0])
            ez = ez_temp - np.dot(ez_temp, ex) * ex
            ez = ez / np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        T = np.zeros((12, 12))
        R = np.column_stack((ex, ey, ez))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        dof_i = slice(6 * node_i, 6 * node_i + 6)
        dof_j = slice(6 * node_j, 6 * node_j + 6)
        u_elem = np.concatenate([u[dof_i], u[dof_j]])
        f_elem = T @ (ke @ (T.T @ u_elem))
        P_axial = f_elem[0]
        kg = np.zeros((12, 12))
        kg[1, 1] = kg[7, 7] = 6 / (5 * L)
        kg[1, 7] = kg[7, 1] = -6 / (5 * L)
        kg[2, 2] = kg[8, 8] = 6 / (5 * L)
        kg[2, 8] = kg[8, 2] = -6 / (5 * L)
        kg[4, 4] = kg[10, 10] = 2 * L / 15
        kg[4, 10] = kg[10, 4] = -L / 30
        kg[5, 5] = kg[11, 11] = 2 * L / 15
        kg[5, 11] = kg[11, 5] = -L / 30
        kg *= -P_axial
        kg_global = T.T @ kg @ T
        dofs = np.r_[dof_i, dof_j]
        Kg[np.ix_(dofs, dofs)] += kg_global
    Kgff = Kg[np.ix_(free_dofs, free_dofs)]
    (eigvals, eigvecs) = scipy.linalg.eigh(Kff, -Kgff)
    pos_eigs = eigvals[eigvals > 0]
    if len(pos_eigs) == 0:
        raise ValueError('No positive eigenvalues found')
    critical_load_factor = pos_eigs[0]
    critical_mode_reduced = eigvecs[:, eigvals > 0][:, 0]
    critical_mode = np.zeros(n_dof)
    critical_mode[free_dofs] = critical_mode_reduced
    return (critical_load_factor, critical_mode)