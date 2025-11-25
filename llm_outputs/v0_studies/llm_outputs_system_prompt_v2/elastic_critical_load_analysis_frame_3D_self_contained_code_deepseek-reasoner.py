def elastic_critical_load_analysis_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    dof_map = {}
    for i in range(n_nodes):
        dof_map[i] = list(range(6 * i, 6 * i + 6))
    constrained_dofs = set()
    for (node_idx, bc) in boundary_conditions.items():
        if all((isinstance(x, bool) for x in bc)):
            for (dof_local, is_fixed) in enumerate(bc):
                if is_fixed:
                    constrained_dofs.add(6 * node_idx + dof_local)
        else:
            for dof_local in bc:
                constrained_dofs.add(6 * node_idx + dof_local)
    free_dofs = sorted(set(range(n_dofs)) - constrained_dofs)
    n_free = len(free_dofs)
    K_global = np.zeros((n_dofs, n_dofs))
    P_global = np.zeros(n_dofs)
    for elem in elements:
        i = elem['node_i']
        j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        Iy = elem['Iy']
        Iz = elem['Iz']
        J = elem['J']
        I_rho = elem['I_rho']
        L = np.linalg.norm(node_coords[j] - node_coords[i])
        Ke_local = np.zeros((12, 12))
        Ke_local[0, 0] = E * A / L
        Ke_local[0, 6] = -E * A / L
        Ke_local[6, 0] = -E * A / L
        Ke_local[6, 6] = E * A / L
        G = E / (2 * (1 + nu))
        phi_y = 12 * E * Iz / (G * A * L * L) if A > 0 else 0
        phi_z = 12 * E * Iy / (G * A * L * L) if A > 0 else 0
        bending_y_terms = E * Iz / (L * L * L * (1 + phi_y))
        bending_z_terms = E * Iy / (L * L * L * (1 + phi_z))
        Ke_local[1, 1] = 12 * bending_y_terms
        Ke_local[1, 5] = 6 * L * bending_y_terms
        Ke_local[1, 7] = -12 * bending_y_terms
        Ke_local[1, 11] = 6 * L * bending_y_terms
        Ke_local[5, 1] = 6 * L * bending_y_terms
        Ke_local[5, 5] = (4 + phi_y) * L * L * bending_y_terms
        Ke_local[5, 7] = -6 * L * bending_y_terms
        Ke_local[5, 11] = (2 - phi_y) * L * L * bending_y_terms
        Ke_local[7, 1] = -12 * bending_y_terms
        Ke_local[7, 5] = -6 * L * bending_y_terms
        Ke_local[7, 7] = 12 * bending_y_terms
        Ke_local[7, 11] = -6 * L * bending_y_terms
        Ke_local[11, 1] = 6 * L * bending_y_terms
        Ke_local[11, 5] = (2 - phi_y) * L * L * bending_y_terms
        Ke_local[11, 7] = -6 * L * bending_y_terms
        Ke_local[11, 11] = (4 + phi_y) * L * L * bending_y_terms
        Ke_local[2, 2] = 12 * bending_z_terms
        Ke_local[2, 4] = -6 * L * bending_z_terms
        Ke_local[2, 8] = -12 * bending_z_terms
        Ke_local[2, 10] = -6 * L * bending_z_terms
        Ke_local[4, 2] = -6 * L * bending_z_terms
        Ke_local[4, 4] = (4 + phi_z) * L * L * bending_z_terms
        Ke_local[4, 8] = 6 * L * bending_z_terms
        Ke_local[4, 10] = (2 - phi_z) * L * L * bending_z_terms
        Ke_local[8, 2] = -12 * bending_z_terms
        Ke_local[8, 4] = 6 * L * bending_z_terms
        Ke_local[8, 8] = 12 * bending_z_terms
        Ke_local[8, 10] = 6 * L * bending_z_terms
        Ke_local[10, 2] = -6 * L * bending_z_terms
        Ke_local[10, 4] = (2 - phi_z) * L * L * bending_z_terms
        Ke_local[10, 8] = 6 * L * bending_z_terms
        Ke_local[10, 10] = (4 + phi_z) * L * L * bending_z_terms
        Ke_local[3, 3] = G * J / L
        Ke_local[3, 9] = -G * J / L
        Ke_local[9, 3] = -G * J / L
        Ke_local[9, 9] = G * J / L
        x_axis = node_coords[j] - node_coords[i]
        x_axis = x_axis / np.linalg.norm(x_axis)
        if elem['local_z'] is not None:
            z_axis = np.array(elem['local_z'])
            z_axis = z_axis / np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis)
        else:
            if abs(x_axis[2]) > 0.9:
                y_axis = np.array([0.0, 1.0, 0.0])
            else:
                y_axis = np.array([0.0, 0.0, 1.0])
            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_axis)
        R = np.zeros((3, 3))
        R[:, 0] = x_axis
        R[:, 1] = y_axis
        R[:, 2] = z_axis
        T = np.zeros((12, 12))
        for i_block in range(4):
            start_row = 3 * i_block
            T[start_row:start_row + 3, start_row:start_row + 3] = R
        Ke_global_elem = T.T @ Ke_local @ T
        dofs_elem = dof_map[i] + dof_map[j]
        for (idx_i, dof_i) in enumerate(dofs_elem):
            for (idx_j, dof_j) in enumerate(dofs_elem):
                K_global[dof_i, dof_j] += Ke_global_elem[idx_i, idx_j]
    for (node_idx, load) in nodal_loads.items():
        start_dof = 6 * node_idx
        for (dof_local, value) in enumerate(load):
            P_global[start_dof + dof_local] += value
    K_free = K_global[np.ix_(free_dofs, free_dofs)]
    P_free = P_global[free_dofs]
    try:
        u_free = scipy.linalg.solve(K_free, P_free, assume_a='sym')
    except scipy.linalg.LinAlgError:
        raise ValueError('Singular stiffness matrix after applying boundary conditions')
    u_full = np.zeros(n_dofs)
    u_full[free_dofs] = u_free
    Kg_global = np.zeros((n_dofs, n_dofs))
    for elem in elements:
        i = elem['node_i']
        j = elem['node_j']
        E = elem['E']
        A = elem['A']
        Iy = elem['Iy']
        Iz = elem['Iz']
        I_rho = elem['I_rho']
        L = np.linalg.norm(node_coords[j] - node_coords[i])
        dofs_elem = dof_map[i] + dof_map[j]
        u_elem = u_full[dofs_elem]
        x_axis = node_coords[j] - node_coords[i]
        x_axis = x_axis / np.linalg.norm(x_axis)
        if elem['local_z'] is not None:
            z_axis = np.array(elem['local_z'])
            z_axis = z_axis / np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis)
        else:
            if abs(x_axis[2]) > 0.9:
                y_axis = np.array([0.0, 1.0, 0.0])
            else:
                y_axis = np.array([0.0, 0.0, 1.0])
            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_axis)
        R = np.zeros((3, 3))
        R[:, 0] = x_axis
        R[:, 1] = y_axis
        R[:, 2] = z_axis
        T = np.zeros((12, 12))
        for i_block in range(4):
            start_row = 3 * i_block
            T[start_row:start_row + 3, start_row:start_row + 3] = R
        u_local = T @ u_elem
        axial_force = E * A / L * (u_local[6] - u_local[0])
        shear_y = 6 * E * Iz / (L * L * (1 + phi_y)) * (u_local[5] + u_local[11] - 2 / L * (u_local[7] - u_local[1]))
        shear_z = 6 * E * Iy / (L * L * (1 + phi_z)) * (-u_local[4] - u_local[10] - 2 / L * (u_local[8] - u_local[2]))
        torsion = G * J / L * (u_local[9] - u_local[3])
        moment_y = 2 * E * Iz / (L * (1 + phi_y)) * (2 * u_local[5] + u_local[11] - 3 / L * (u_local[7] - u_local[1]))
        moment_z = 2 * E * Iy / (L * (1 + phi_z)) * (2 * u_local[4] + u_local[10] + 3 / L * (u_local[8] - u_local[2]))
        Kg_local = np.zeros((12, 12))
        axial_term = axial_force / L
        Kg_local[0, 0] = axial_term
        Kg_local[0, 6] = -axial_term
        Kg_local[6, 0] = -axial_term
        Kg_local[6, 6] = axial_term
        bending_y_term = 6 * moment_y / (5 * L)
        Kg_local[1, 1] = bending_y_term
        Kg_local[1, 5] = bending_y_term / 10
        Kg_local[1, 7] = -bending_y_term
        Kg_local[1, 11] = bending_y_term / 10
        Kg_local[5, 1] = bending_y_term / 10
        Kg_local[5, 5] = 2 * L * bending_y_term / 15
        Kg_local[5, 7] = -bending_y_term / 10
        Kg_local[5, 11] = -L * bending_y_term / 30
        Kg_local[7, 1] = -bending_y_term
        Kg_local[7, 5] = -bending_y_term / 10
        Kg_local[7, 7] = bending_y_term
        Kg_local[7, 11] = -bending_y_term / 10
        Kg_local[11, 1] = bending_y_term / 10
        Kg_local[11, 5] = -L * bending_y_term / 30
        Kg_local[11, 7] = -bending_y_term / 10
        Kg_local[11, 11] = 2 * L * bending_y_term / 15
        bending_z_term = 6 * moment_z / (5 * L)
        Kg_local[2, 2] = bending_z_term
        Kg_local[2, 4] = -bending_z_term / 10
        Kg_local[2, 8] = -bending_z_term
        Kg_local[2, 10] = -bending_z_term / 10
        Kg_local[4, 2] = -bending_z_term / 10
        Kg_local[4, 4] = 2 * L * bending_z_term / 15
        Kg_local[4, 8] = bending_z_term / 10
        Kg_local[4, 10] = -L * bending_z_term / 30
        Kg_local[8, 2] = -bending_z_term
        Kg_local[8, 4] = bending_z_term / 10
        Kg_local[8, 8] = bending_z_term
        Kg_local[8, 10] = bending_z_term / 10
        Kg_local[10, 2] = -bending_z_term / 10
        Kg_local[10, 4] = -L * bending_z_term / 30
        Kg_local[10, 8] = bending_z_term / 10
        Kg_local[10, 10] = 2 * L * bending_z_term / 15
        Kg_global_elem = T.T @ Kg_local @ T
        for (idx_i, dof_i) in enumerate(dofs_elem):
            for (idx_j, dof_j) in enumerate(dofs_elem):
                Kg_global[dof_i, dof_j] += Kg_global_elem[idx_i, idx_j]
    Kg_free = Kg_global[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_free, -Kg_free)
    except (scipy.linalg.LinAlgError, ValueError):
        raise ValueError('Generalized eigenproblem solution failed')
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalues found')
    min_eigenvalue_idx = np.argmin(positive_eigenvalues)
    critical_load_factor = positive_eigenvalues[min_eigenvalue_idx]
    corresponding_eigenvector_idx = np.where(eigenvalues == positive_eigenvalues[min_eigenvalue_idx])[0][0]
    mode_free = eigenvectors[:, corresponding_eigenvector_idx]
    deformed_shape_vector = np.zeros(n_dofs)
    deformed_shape_vector[free_dofs] = mode_free
    return (critical_load_factor, deformed_shape_vector)