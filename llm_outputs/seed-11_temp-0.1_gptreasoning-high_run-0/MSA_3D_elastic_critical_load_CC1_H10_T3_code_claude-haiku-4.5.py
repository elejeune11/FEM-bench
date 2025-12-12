def MSA_3D_elastic_critical_load_CC1_H10_T3(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    """
    Perform linear (eigenvalue) buckling analysis for a 3D frame and return the
    elastic critical load factor and associated global buckling mode shape.
    Overview
    --------
    The routine:
      1) Assembles the global elastic stiffness matrix `K`.
      2) Assembles the global reference load vector `P`.
      3) Solves the linear static problem `K u = P` (with boundary conditions) to
         obtain the displacement state `u` under the reference load.
      4) Assembles the geometric stiffness `K_g` consistent with that state.
      5) Solves the generalized eigenproblem on the free DOFs,
             K φ = -λ K_g φ,
         and returns the smallest positive eigenvalue `λ` as the elastic
         critical load factor and its corresponding global mode shape `φ`
         (constrained DOFs set to zero).
    Parameters
    ----------
    node_coords : (N, 3) float ndarray
        Global coordinates of the N nodes (row i → [x, y, z] of node i, 0-based).
    elements : iterable of dict
        Each dict must contain:
            'node_i', 'node_j' : int
                End node indices (0-based).
            'E', 'nu', 'A', 'I_y', 'I_z', 'J' : float
                Material and geometric properties.
            'local_z' : array-like of shape (3,), optional
                Unit vector in global coordinates defining the local z-axis orientation.
                Must not be parallel to the beam axis. If None, a default is chosen.
                Default local_z = global z, unless the beam lies along global z — then default local_z = global y.
    boundary_conditions : dict[int, Sequence[int]]
        Maps node index to a 6-element iterable of 0 (free) or 1 (fixed) values.
        Omitted nodes are assumed to have all DOFs free.
    nodal_loads : dict[int, Sequence[float]]
        Maps node index to a 6-element array of applied loads:
        [Fx, Fy, Fz, Mx, My, Mz]. Omitted nodes are assumed to have zero loads.
    Returns
    -------
    elastic_critical_load_factor : float
        The smallest positive eigenvalue `λ` (> 0). If `P` is the reference load
        used to form `K_g`, then the predicted elastic buckling load is
        `P_cr = λ · P`.
    deformed_shape_vector : (6*n_nodes,) ndarray of float
        Global buckling mode vector with constrained DOFs set to zero. No
        normalization is applied (mode scale is arbitrary; only the shape matters).
    Assumptions
    -----------
      `[u_x, u_y, u_z, θ_x, θ_y, θ_z]`.
      represented via `K_g` assembled at the reference load state, not via a full
      nonlinear equilibrium/path-following analysis.
    Raises
    ------
    ValueError
        Propagated from called routines if:
    Notes
    -----
      if desired (e.g., by max absolute translational DOF).
      the returned mode can depend on numerical details.
        + **Tension (+Fx2)** increases lateral/torsional stiffness.
        + Compression (-Fx2) decreases it and may trigger buckling when K_e + K_g becomes singular.
    """
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    K_global = np.zeros((n_dof, n_dof))
    K_g_global = np.zeros((n_dof, n_dof))
    P_global = np.zeros(n_dof)
    for (node_idx, loads) in nodal_loads.items():
        for dof_idx in range(6):
            P_global[6 * node_idx + dof_idx] = loads[dof_idx]
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        xi = node_coords[node_i]
        xj = node_coords[node_j]
        dx = xj - xi
        L = np.linalg.norm(dx)
        ex = dx / L
        local_z = elem.get('local_z', None)
        if local_z is None:
            if abs(ex[2]) < 0.99:
                local_z = np.array([0.0, 0.0, 1.0])
            else:
                local_z = np.array([0.0, 1.0, 0.0])
        else:
            local_z = np.array(local_z) / np.linalg.norm(local_z)
        ey_temp = local_z - np.dot(local_z, ex) * ex
        ey = ey_temp / np.linalg.norm(ey_temp)
        ez = np.cross(ex, ey)
        T = np.zeros((12, 12))
        for i in range(2):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = np.array([ex, ey, ez])
            T[6 + 3 * i:6 + 3 * i + 3, 6 + 3 * i:6 + 3 * i + 3] = np.array([ex, ey, ez])
        G = E / (2 * (1 + nu))
        k_local = np.zeros((12, 12))
        k_local[0, 0] = E * A / L
        k_local[0, 6] = -E * A / L
        k_local[6, 0] = -E * A / L
        k_local[6, 6] = E * A / L
        k_local[3, 3] = G * J / L
        k_local[3, 9] = -G * J / L
        k_local[9, 3] = -G * J / L
        k_local[9, 9] = G * J / L
        k_local[2, 2] = 12 * E * I_z / L ** 3
        k_local[2, 4] = 6 * E * I_z / L ** 2
        k_local[2, 8] = -12 * E * I_z / L ** 3
        k_local[2, 10] = 6 * E * I_z / L ** 2
        k_local[4, 2] = 6 * E * I_z / L ** 2
        k_local[4, 4] = 4 * E * I_z / L
        k_local[4, 8] = -6 * E * I_z / L ** 2
        k_local[4, 10] = 2 * E * I_z / L
        k_local[8, 2] = -12 * E * I_z / L ** 3
        k_local[8, 4] = -6 * E * I_z / L ** 2
        k_local[8, 8] = 12 * E * I_z / L ** 3
        k_local[8, 10] = -6 * E * I_z / L ** 2
        k_local[10, 2] = 6 * E * I_z / L ** 2
        k_local[10, 4] = 2 * E * I_z / L
        k_local[10, 8] = -6 * E * I_z / L ** 2
        k_local[10, 10] = 4 * E * I_z / L
        k_local[1, 1] = 12 * E * I_y / L ** 3
        k_local[1, 5] = -6 * E * I_y / L ** 2
        k_local[1, 7] = -12 * E * I_y / L ** 3
        k_local[1, 11] = -6 * E * I_y / L ** 2
        k_local[5, 1] = -6 * E * I_y / L ** 2
        k_local[5, 5] = 4 * E * I_y / L
        k_local[5, 7] = 6 * E * I_y / L ** 2
        k_local[5, 11] = 2 * E * I_y / L
        k_local[7, 1] = -12 * E * I_y / L ** 3
        k_local[7, 5] = 6 * E * I_y / L ** 2
        k_local[7, 7] = 12 * E * I_y / L ** 3
        k_local[7, 11] = 6 * E * I_y / L ** 2
        k_local[11, 1] = -6 * E * I_y / L ** 2
        k_local[11, 5] = 2 * E * I_y / L
        k_local[11, 7] = 6 * E * I_y / L ** 2
        k_local[11, 11] = 4 * E * I_y / L
        k_global = T.T @ k_local @ T
        dof_indices = np.array([6 * node_i, 6 * node_i + 1, 6 * node_i + 2, 6 * node_i + 3, 6 * node_i + 4, 6 * node_i + 5, 6 * node_j, 6 * node_j + 1, 6 * node_j + 2, 6 * node_j + 3, 6 * node_j + 4, 6 * node_j + 5])
        for i in range(12):
            for j in range(12):
                K_global[dof_indices[i], dof_indices[j]] += k_global[i, j]
    free_dofs = []
    for dof in range(n_dof):
        node_idx = dof // 6
        local_dof = dof % 6
        if node_idx not in boundary_conditions or boundary_conditions[node_idx][local_dof] == 0:
            free_dofs.append(dof)
    free_dofs = np.array(free_dofs)
    constrained_dofs = np.array([dof for dof in range(n_dof) if dof not in free_dofs])
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    P_f = P_global[free_dofs]
    try:
        u_f = np.linalg.solve(K_ff, P_f)
    except np.linalg.LinAlgError:
        raise ValueError('Stiffness matrix is singular; check boundary conditions')
    u_global = np.zeros(n_dof)
    u_global[free_dofs] = u_f
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        xi = node_coords[node_i]
        xj = node_coords[node_j]
        dx = xj - xi
        L = np.linalg.norm(dx)
        ex = dx / L
        local_z = elem.get('local_z', None)
        if local_z is None:
            if abs(ex[2]) < 0.99:
                local_z = np.array([0.0, 0.0, 1.0])
            else:
                local_z = np.array([0.0, 1.0, 0.0])
        else:
            local_z = np.array(local_z) / np.linalg.norm(local_z)
        ey_temp = local_z - np.dot(local_z, ex) * ex
        ey = ey_temp / np.linalg.norm(ey_temp)
        ez = np.cross(ex, ey)
        T = np.zeros((12, 12))
        for i in range(2):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = np.array([ex, ey, ez])
            T[6 + 3 * i:6 + 3 * i + 3, 6 + 3 * i:6 + 3 * i + 3] = np.array([ex, ey, ez])
        u_elem_global = np.concatenate([u_global[6 * node_i:6 * node_i + 6], u_global[6 * node_j:6 * node_j + 6]])
        u_elem_local = T @ u_elem_global
        G = E / (2 * (1 + nu))
        Fx1 = E * A / L * (u_elem_local[6] - u_elem_local[0])
        Fx2 = -Fx1
        Mx1 = G * J / L * (u_elem_local[9] - u_elem_local[3])
        Mx2 = -Mx1
        My1 = 4 * E * I_z / L * u_elem_local[4] + 2 * E * I_z / L * u_elem_local[10] + 6 * E * I_z / L ** 2 * (u_elem_local[8] - u_elem_local[2])
        My2 = 2 * E * I_z / L * u_elem_local[4] + 4 * E * I_z / L * u_elem_local[10] + 6 * E * I_z / L ** 2 * (u_elem_local[2] - u_elem_local[8])
        Mz1 = 4 * E * I_y / L * u_elem_local[5] + 2 * E * I_y / L * u_elem_local[11] - 6 * E * I_y / L ** 2 * (u_elem_local[7] - u_elem_local[1])
        Mz2 = 2 * E * I_y / L * u_elem_local[5] + 4 * E * I_y / L * u_elem_local[11] - 6 * E * I_y / L ** 2 * (u_elem_local[1] - u_elem_local[7])
        kg_local = np.zeros((12, 12))
        kg_local[1, 1] = Fx2 / L
        kg_local[1, 7] = -Fx2 / L
        kg_local[2, 2] = Fx2 / L
        kg_local[2, 8] = -Fx2 / L
        kg_local[7, 1] = -Fx2 / L
        kg_local[7, 7] = Fx2 / L
        kg_local[8, 2] = -Fx2 / L
        kg_local[8, 8] = Fx2 / L
        kg_local[4, 4] = Fx2 / L
        kg_local[4, 10] = -Fx2 / L
        kg_local[5, 5] = Fx2 / L
        kg_local[5, 11] = -Fx2 / L
        kg_local[10, 4] = -Fx2 / L
        kg_local[10, 10] = Fx2 / L
        kg_local[11, 5] = -Fx2 / L
        kg_local[11, 11] = Fx2 / L
        kg_local[1, 4] = -Mz2 / L
        kg_local[1, 10] = -Mz2 / L
        kg_local[2, 5] = My2 / L
        kg_local[2, 11] = My2 / L
        kg_local[4, 1] = -Mz2 / L
        kg_local[4, 7] = Mz2 / L
        kg_local[5, 2] = My2 / L
        kg_local[5, 8] = -My2 / L
        kg_local[7, 4] = Mz2 / L
        kg_local[7, 10] = Mz2 / L
        kg_local[8, 5] = -My2 / L
        kg_local[8, 11] = -My2 / L
        kg_local[10, 1] = -Mz2 / L
        kg_local[10, 7] = Mz2 / L
        kg_local[11, 2] = My2 / L
        kg_local[11, 8] = -My2 / L
        kg_global = T.T @ kg_local @ T
        dof_indices = np.array([6 * node_i, 6 * node_i + 1, 6 * node_i + 2, 6 * node_i + 3, 6 * node_i + 4, 6 * node_i + 5, 6 * node_j, 6 * node_j + 1, 6 * node_j + 2, 6 * node_j + 3, 6 * node_j + 4, 6 * node_j + 5])
        for i in range(12):
            for j in range(12):
                K_g_global[dof_indices[i], dof_indices[j]] += kg_global[i, j]
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_g_ff = K_g_global[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_ff, -K_g_ff)
    except np.linalg.LinAlgError:
        raise ValueError('Eigenvalue problem failed; check matrices')
    positive_eigs = eigenvalues[eigenvalues > 1e-10]
    if len(positive_eigs) == 0:
        raise ValueError('No positive eigenvalue found')
    lambda_cr = positive_eigs[0]
    idx = np.where(eigenvalues == lambda_cr)[0][0]
    mode_f = eigenvectors[:, idx]
    mode_global = np.zeros(n_dof)
    mode_global[free_dofs] = mode_f
    return (float(lambda_cr), mode_global)