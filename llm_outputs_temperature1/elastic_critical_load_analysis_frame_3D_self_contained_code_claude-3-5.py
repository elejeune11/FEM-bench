def elastic_critical_load_analysis_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    bc_bool = np.zeros(n_dof, dtype=bool)
    for (node_idx, dofs) in boundary_conditions.items():
        node_start = 6 * node_idx
        if all((isinstance(x, bool) for x in dofs)):
            bc_bool[node_start:node_start + 6] = dofs
        else:
            bc_bool[node_start + np.array(dofs)] = True
    free_dofs = ~bc_bool
    K = np.zeros((n_dof, n_dof))
    Kg = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    for (node_idx, loads) in nodal_loads.items():
        P[6 * node_idx:6 * node_idx + 6] = loads
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        xi = node_coords[i]
        xj = node_coords[j]
        L = np.linalg.norm(xj - xi)
        ex = (xj - xi) / L
        if elem['local_z'] is not None:
            ez = np.array(elem['local_z'])
            ez = ez / np.linalg.norm(ez)
        else:
            global_z = np.array([0, 0, 1])
            ez = global_z - np.dot(global_z, ex) * ex
            if np.allclose(ez, 0):
                ez = np.array([0, 1, 0])
            ez = ez / np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        T = np.zeros((3, 3))
        T[0] = ex
        T[1] = ey
        T[2] = ez
        R = np.zeros((12, 12))
        for k in range(4):
            R[3 * k:3 * k + 3, 3 * k:3 * k + 3] = T
        E = elem['E']
        A = elem['A']
        Iy = elem['Iy']
        Iz = elem['Iz']
        J = elem['J']
        nu = elem['nu']
        EA_L = E * A / L
        EIy_L3 = E * Iy / L ** 3
        EIz_L3 = E * Iz / L ** 3
        GJ_L = E / (2 * (1 + nu)) * J / L
        ke = np.zeros((12, 12))
        ke[0, 0] = ke[6, 6] = EA_L
        ke[0, 6] = ke[6, 0] = -EA_L
        ke[3, 3] = ke[9, 9] = GJ_L
        ke[3, 9] = ke[9, 3] = -GJ_L
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
        ke = R.T @ ke @ R
        dofs = np.concatenate([np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)])
        for ii in range(12):
            for jj in range(12):
                K[dofs[ii], dofs[jj]] += ke[ii, jj]
    u = np.zeros(n_dof)
    u[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], P[free_dofs])
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        dofs = np.concatenate([np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)])
        u_e = u[dofs]
        kge = np.zeros((12, 12))
        L = np.linalg.norm(node_coords[j] - node_coords[i])
        N = elem['E'] * elem['A'] * (u_e[6] - u_e[0]) / L
        kge[1:3, 1:3] = kge[7:9, 7:9] = N / L * np.eye(2)
        kge[1:3, 7:9] = kge[7:9, 1:3] = -N / L * np.eye(2)
        ex = (node_coords[j] - node_coords[i]) / L
        if elem['local_z'] is not None:
            ez = np.array(elem['local_z'])
            ez = ez / np.linalg.norm(ez)
        else:
            global_z = np.array([0, 0, 1])
            ez = global_z - np.dot(global_z, ex) * ex
            if np.allclose(ez, 0):
                ez = np.array([0, 1, 0])
            ez = ez / np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        T = np.zeros((3, 3))
        T[0] = ex
        T[1] = ey
        T[2] = ez
        R = np.zeros((12, 12))
        for k in range(4):
            R[3 * k:3 * k + 3, 3 * k:3 * k + 3] = T
        kge = R.T @ kge @ R
        for ii in range(12):
            for jj in range(12):
                Kg[dofs[ii], dofs[jj]] += kge[ii, jj]
    K_red = K[np.ix_(free_dofs, free_dofs)]
    Kg_red = Kg[np.ix_(free_dofs, free_dofs)]
    (eigenvals, eigenvecs) = scipy.linalg.eigh(K_red, -Kg_red)
    pos_eigenvals = eigenvals[eigenvals > 0]
    if len(pos_eigenvals) == 0:
        raise ValueError('No positive eigenvalues found')
    lambda_cr = np.min(pos_eigenvals)
    mode_idx = np.where(eigenvals == lambda_cr)[0][0]
    mode_shape = np.zeros(n_dof)
    mode_shape[free_dofs] = eigenvecs[:, mode_idx]
    return (lambda_cr, mode_shape)