def elastic_critical_load_analysis_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    Kg = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    constrained_dofs = set()
    for (node_idx, bc_spec) in boundary_conditions.items():
        node_dofs = range(6 * node_idx, 6 * (node_idx + 1))
        if isinstance(bc_spec[0], bool):
            constrained_dofs.update((dof for (dof, is_fixed) in zip(node_dofs, bc_spec) if is_fixed))
        else:
            constrained_dofs.update((node_dofs[i] for i in bc_spec))
    free_dofs = list(set(range(n_dof)) - constrained_dofs)
    for (node_idx, loads) in nodal_loads.items():
        P[6 * node_idx:6 * (node_idx + 1)] = loads
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
            global_Z = np.array([0, 0, 1])
            if abs(np.dot(ex, global_Z)) < 0.99:
                ez_temp = global_Z
            else:
                ez_temp = np.array([1, 0, 0])
            ez = np.cross(ex, ez_temp)
            ez = ez / np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        T = np.zeros((3, 3))
        T[0] = ex
        T[1] = ey
        T[2] = ez
        Te = np.block([[T, np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))], [np.zeros((3, 3)), T, np.zeros((3, 3)), np.zeros((3, 3))], [np.zeros((3, 3)), np.zeros((3, 3)), T, np.zeros((3, 3))], [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), T]])
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
        ke_global = Te.T @ ke @ Te
        I_rho = elem['I_rho']
        kg = np.zeros((12, 12))
        dofs_i = slice(6 * node_i, 6 * node_i + 6)
        dofs_j = slice(6 * node_j, 6 * node_j + 6)
        elem_dofs = np.r_[dofs_i, dofs_j]
        K[np.ix_(elem_dofs, elem_dofs)] += ke_global
        Kg[np.ix_(elem_dofs, elem_dofs)] += kg
    K_free = K[np.ix_(free_dofs, free_dofs)]
    P_free = P[free_dofs]
    u_free = np.linalg.solve(K_free, P_free)
    u_full = np.zeros(n_dof)
    u_full[free_dofs] = u_free
    K_free = K[np.ix_(free_dofs, free_dofs)]
    Kg_free = Kg[np.ix_(free_dofs, free_dofs)]
    (eigenvals, eigenvecs) = scipy.linalg.eigh(K_free, -Kg_free)
    pos_eigenvals = eigenvals[eigenvals > 0]
    if len(pos_eigenvals) == 0:
        raise ValueError('No positive eigenvalues found')
    critical_load_factor = np.min(pos_eigenvals)
    mode_index = np.where(eigenvals == critical_load_factor)[0][0]
    mode_shape = np.zeros(n_dof)
    mode_shape[free_dofs] = eigenvecs[:, mode_index]
    return (critical_load_factor, mode_shape)