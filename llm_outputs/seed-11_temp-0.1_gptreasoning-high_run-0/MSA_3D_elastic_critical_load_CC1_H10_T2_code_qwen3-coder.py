def MSA_3D_elastic_critical_load_CC1_H10_T2(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
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
    Helper Functions (used here)
    ----------------------------
        Returns the 12x12 local geometric stiffness matrix with torsion-bending coupling for a 3D Euler-Bernoulli beam element.
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
    K = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    K_g = np.zeros((n_dof, n_dof))

    def node_dofs(node_idx):
        return slice(6 * node_idx, 6 * (node_idx + 1))
    for (node_idx, load) in nodal_loads.items():
        P[node_dofs(node_idx)] += np.array(load)
    element_internal_forces = []
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        Iy = elem['I_y']
        Iz = elem['I_z']
        J = elem['J']
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if L == 0:
            raise ValueError('Zero-length element detected.')
        (lx, ly, lz) = (dx / L, dy / L, dz / L)
        if 'local_z' in elem and elem['local_z'] is not None:
            local_z = np.array(elem['local_z'])
        else:
            global_z = np.array([0.0, 0.0, 1.0])
            beam_axis = np.array([lx, ly, lz])
            if np.allclose(np.cross(beam_axis, global_z), 0):
                local_z = np.array([0.0, 1.0, 0.0])
            else:
                local_z = global_z
        local_y = np.cross([lx, ly, lz], local_z)
        local_y /= np.linalg.norm(local_y)
        local_z = np.cross([lx, ly, lz], local_y)
        local_z /= np.linalg.norm(local_z)
        T = np.zeros((12, 12))
        R = np.array([[lx, ly, lz], [local_y[0], local_y[1], local_y[2]], [local_z[0], local_z[1], local_z[2]]])
        for i in range(4):
            T[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = R
        EAL = E * A / L
        EIy_L3 = 12 * E * Iy / L ** 3
        EIy_L2 = 6 * E * Iy / L ** 2
        EIy_L = E * Iy / L
        EIz_L3 = 12 * E * Iz / L ** 3
        EIz_L2 = 6 * E * Iz / L ** 2
        EIz_L = E * Iz / L
        GJ_L = E / (2 * (1 + nu)) * J / L
        k_local = np.zeros((12, 12))
        k_local[0, 0] = EAL
        k_local[0, 6] = -EAL
        k_local[6, 6] = EAL
        k_local[3, 3] = GJ_L
        k_local[3, 9] = -GJ_L
        k_local[9, 9] = GJ_L
        k_local[1, 1] = EIz_L3
        k_local[1, 5] = EIz_L2
        k_local[1, 7] = -EIz_L3
        k_local[1, 11] = EIz_L2
        k_local[5, 5] = 4 * EIz_L
        k_local[5, 7] = -EIz_L2
        k_local[5, 11] = 2 * EIz_L
        k_local[7, 7] = EIz_L3
        k_local[7, 11] = -EIz_L2
        k_local[11, 11] = 4 * EIz_L
        k_local[2, 2] = EIy_L3
        k_local[2, 4] = -EIy_L2
        k_local[2, 8] = -EIy_L3
        k_local[2, 10] = -EIy_L2
        k_local[4, 4] = 4 * EIy_L
        k_local[4, 8] = EIy_L2
        k_local[4, 10] = 2 * EIy_L
        k_local[8, 8] = EIy_L3
        k_local[8, 10] = EIy_L2
        k_local[10, 10] = 4 * EIy_L
        k_local = k_local + k_local.T - np.diag(np.diag(k_local))
        k_global = T.T @ k_local @ T
        dofs_i = node_dofs(node_i)
        dofs_j = node_dofs(node_j)
        global_dofs = np.concatenate([np.arange(dofs_i.start, dofs_i.stop), np.arange(dofs_j.start, dofs_j.stop)])
        for (idx_i, dof_i) in enumerate(global_dofs):
            for (idx_j, dof_j) in enumerate(global_dofs):
                K[dof_i, dof_j] += k_global[idx_i, idx_j]
    fixed_dofs = []
    for (node_idx, constraints) in boundary_conditions.items():
        for (i, is_fixed) in enumerate(constraints):
            if is_fixed:
                fixed_dofs.append(6 * node_idx + i)
    free_dofs = list(set(range(n_dof)) - set(fixed_dofs))
    free_dofs.sort()
    if len(free_dofs) == 0:
        raise ValueError('All DOFs are fixed. No free DOFs for analysis.')
    K_free = K[np.ix_(free_dofs, free_dofs)]
    P_free = P[free_dofs]
    try:
        u_free = np.linalg.solve(K_free, P_free)
    except np.linalg.LinAlgError:
        raise ValueError('Singular stiffness matrix. Check boundary conditions.')
    u = np.zeros(n_dof)
    u[free_dofs] = u_free
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        E = elem['E']
        A = elem['A']
        Iy = elem['I_y']
        Iz = elem['I_z']
        J = elem['J']
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        (lx, ly, lz) = (dx / L, dy / L, dz / L)
        if 'local_z' in elem and elem['local_z'] is not None:
            local_z = np.array(elem['local_z'])
        else:
            global_z = np.array([0.0, 0.0, 1.0])
            beam_axis = np.array([lx, ly, lz])
            if np.allclose(np.cross(beam_axis, global_z), 0):
                local_z = np.array([0.0, 1.0, 0.0])
            else:
                local_z = global_z
        local_y = np.cross([lx, ly, lz], local_z)
        local_y /= np.linalg.norm(local_y)
        local_z = np.cross([lx, ly, lz], local_y)
        local_z /= np.linalg.norm(local_z)
        T = np.zeros((12, 12))
        R = np.array([[lx, ly, lz], [local_y[0], local_y[1], local_y[2]], [local_z[0], local_z[1], local_z[2]]])
        for i in range(4):
            T[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = R
        u_elem_global = np.concatenate([u[node_dofs(node_i)], u[node_dofs(node_j)]])
        u_elem_local = T @ u_elem_global
        EAL = E * A / L
        EIy_L3 = 12 * E * Iy / L ** 3
        EIy_L2 = 6 * E * Iy / L ** 2
        EIy_L = E * Iy / L
        EIz_L3 = 12 * E * Iz / L ** 3
        EIz_L2 = 6 * E * Iz / L ** 2
        EIz_L = E * Iz / L
        GJ_L = E / (2 * (1 + 0.3)) * J / L
        Fx2 = EAL * (u_elem_local[6] - u_elem_local[0])
        My1 = -EIz_L2 * u_elem_local[1] - 2 * EIz_L * u_elem_local[5] + EIz_L2 * u_elem_local[7] + EIz_L * u_elem_local[11]
        Mz1 = EIy_L2 * u_elem_local[2] + 2 * EIy_L * u_elem_local[4] - EIy_L2 * u_elem_local[8] - EIy_L * u_elem_local[10]
        My2 = EIz_L2 * u_elem_local[1] + EIz_L * u_elem_local[5] - EIz_L2 * u_elem_local[7] - 2 * EIz_L * u_elem_local[11]
        Mz2 = -EIy_L2 * u_elem_local[2] - EIy_L * u_elem_local[4] + EIy_L2 * u_elem_local[8] + 2 * EIy_L * u_elem_local[10]
        Mx2 = GJ_L * (u_elem_local[9] - u_elem_local[3])
        element_internal_forces.append((L, A, J, Fx2, Mx2, My1, Mz1, My2, Mz2))
    for (idx, elem) in enumerate(elements):
        node_i = elem['node_i']
        node_j = elem['node_j']
        (L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2) = element_internal_forces[idx]
        k_g_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        (lx, ly, lz) = (dx / L, dy / L, dz / L)
        if 'local_z' in elem and elem['local_z'] is not None:
            local_z = np.array(elem['local_z'])
        else:
            global_z = np.array([0.0, 0.0, 1.0])
            beam_axis = np.array([lx, ly, lz])
            if np.allclose(np.cross(beam_axis, global_z), 0):
                local_z = np.array([0.0, 1.0, 0.0])
            else:
                local_z = global_z
        local_y = np.cross([lx, ly, lz], local_z)
        local_y /= np.linalg.norm(local_y)
        local_z = np.cross([lx, ly, lz], local_y)
        local_z /= np.linalg.norm(local_z)
        T = np.zeros((12, 12))
        R = np.array([[lx, ly, lz], [local_y[0], local_y[1], local_y[2]], [local_z[0], local_z[1], local_z[2]]])
        for i in range(4):
            T[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = R
        k_g_global = T.T @ k_g_local @ T
        dofs_i = node_dofs(node_i)
        dofs_j = node_dofs(node_j)
        global_dofs = np.concatenate([np.arange(dofs_i.start, dofs_i.stop), np.arange(dofs_j.start, dofs_j.stop)])
        for (idx_i, dof_i) in enumerate(global_dofs):
            for (idx_j, dof_j) in enumerate(global_dofs):
                K_g[dof_i, dof_j] += k_g_global[idx_i, idx_j]
    K_free = K[np.ix_(free_dofs, free_dofs)]
    K_g_free = K_g[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eig(K_free, -K_g_free)
    except np.linalg.LinAlgError:
        raise ValueError('Failed to solve eigenvalue problem.')
    real_eigenvalues = []
    real_eigenvectors = []
    for (i, val) in enumerate(eigenvalues):
        if np.isreal(val) and val > 0:
            real_eigenvalues.append(np.real(val))
            real_eigenvectors.append(np.real(eigenvectors[:, i]))
    if not real_eigenvalues:
        raise ValueError('No positive real eigenvalues found.')
    min_idx = np.argmin(real_eigenvalues)
    lambda_min = real_eigenvalues[min_idx]
    phi_free = real_eigenvectors[min_idx]
    phi = np.zeros(n_dof)
    phi[free_dofs] = phi_free
    return (lambda_min, phi)