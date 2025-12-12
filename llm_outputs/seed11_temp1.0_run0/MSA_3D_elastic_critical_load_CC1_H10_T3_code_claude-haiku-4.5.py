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
    for node_idx, loads in nodal_loads.items():
        dof_start = node_idx * 6
        P_global[dof_start:dof_start + 6] = loads
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        coords_i = node_coords[node_i]
        coords_j = node_coords[node_j]
        L_vec = coords_j - coords_i
        L = np.linalg.norm(L_vec)
        local_x = L_vec / L
        if 'local_z' in elem and elem['local_z'] is not None:
            local_z = np.array(elem['local_z'])
            local_z = local_z / np.linalg.norm(local_z)
        else:
            global_z = np.array([0., 0., 1.])
            if abs(np.dot(local_x, global_z)) > 0.99:
                local_z = np.array([0., 1., 0.])
            else:
                local_z = global_z
        local_z = local_z - np.dot(local_z, local_x) * local_x
        local_z = local_z / np.linalg.norm(local_z)
        local_y = np.cross(local_z, local_x)
        local_y = local_y / np.linalg.norm(local_y)
        R = np.zeros((3, 3))
        R[0, :] = local_x
        R[1, :] = local_y
        R[2, :] = local_z
        G = E / (2 * (1 + nu))
        k_local = np.zeros((12, 12))
        k_local[0, 0] = E * A / L
        k_local[0, 6] = -E * A / L
        k_local[6, 0] = -E * A / L
        k_local[6, 6] = E * A / L
        k_local[2, 2] = 12 * E * I_z / L**3
        k_local[2, 4] = 6 * E * I_z / L**2
        k_local[2, 8] = -12 * E * I_z / L**3
        k_local[2, 10] = 6 * E * I_z / L**2
        k_local[4, 2] = 6 * E * I_z / L**2
        k_local[4, 4] = 4 * E * I_z / L
        k_local[4, 8] = -6 * E * I_z / L**2
        k_local[4, 10] = 2 * E * I_z / L
        k_local[8, 2] = -12 * E * I_z / L**3
        k_local[8, 4] = -6 * E * I_z / L**2
        k_local[8, 8] = 12 * E * I_z / L**3
        k_local[8, 10] = -6 * E * I_z / L**2
        k_local[10, 2] = 6 * E * I_z / L**2
        k_local[10, 4] = 2 * E * I_z / L
        k_local[10, 8] = -6 * E * I_z / L**2
        k_local[10, 10] = 4 * E * I_z / L
        k_local[1, 1] = 12 * E * I_y / L**3
        k_local[1, 5] = -6 * E * I_y / L**2
        k_local[1, 7] = -12 * E * I_y / L**3
        k_local[1, 11] = -6 * E * I_y / L**2
        k_local[5, 1] = -6 * E * I_y / L**2
        k_local[5, 5] = 4 * E * I_y / L
        k_local[5, 7] = 6 * E * I_y / L**2
        k_local[5, 11] = 2 * E * I_y / L
        k_local[7, 1] = -12 * E * I_y / L**3
        k_local[7, 5] = 6 * E * I_y / L**2
        k_local[7, 7] = 12 * E * I_y / L**3
        k_local[7, 11] = 6 * E * I_y / L**2
        k_local[11, 1] = -6 * E * I_y / L**2
        k_local[11, 5] = 2 * E * I_y / L
        k_local[11, 7] = 6 * E * I_y / L**2
        k_local[11, 11] = 4 * E * I_y / L
        k_local[3, 3] = G * J / L
        k_local[3, 9] = -G * J / L
        k_local[9, 3] = -G * J / L
        k_local[9, 9] = G * J / L
        T = np.zeros((12, 12))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        k_global = T.T @ k_local @ T
        dof_i = np.array([node_i * 6 + j for j in range(6)])
        dof_j = np.array([node_j * 6 + j for j in range(6)])
        dof_elem = np.concatenate([dof_i, dof_j])
        for ii, di in enumerate(dof_elem):
            for jj, dj in enumerate(dof_elem):
                K_global[di, dj] += k_global[ii, jj]
    fixed_dofs = set()
    for node_idx, bc in boundary_conditions.items():
        for dof_local, is_fixed in enumerate(bc):
            if is_fixed:
                fixed_dofs.add(node_idx * 6 + dof_local)
    free_dofs = np.array([i for i in range(n_dof) if i not in fixed_dofs])
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    P_f = P_global[free_dofs]
    try:
        u_f = np.linalg.solve(K_ff, P_f)
    except np.linalg.LinAlgError:
        raise ValueError("Singular stiffness matrix at free DOFs")
    u_global = np.zeros(n_dof)
    u_global[free_dofs] = u_f
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        coords_i = node_coords[node_i]
        coords_j = node_coords[node_j]
        L_vec = coords_j - coords_i
        L = np.linalg.norm(L_vec)
        local_x = L_vec / L
        if 'local_z' in elem and elem['local_z'] is not None:
            local_z = np.array(elem['local_z'])
            local_z = local_z / np.linalg.norm(local_z)
        else:
            global_z = np.array([0., 0., 1.])
            if abs(np.dot(local_x, global_z)) > 0.99:
                local_z = np.array([0., 1., 0.])
            else:
                local_z = global_z
        local_z = local_z - np.dot(local_z, local_x) * local_x
        local_z = local_z / np.linalg.norm(local_z)
        local_y = np.cross(local_z, local_x)
        local_y = local_y / np.linalg.norm(local_y)
        R = np.zeros((3, 3))
        R[0, :] = local_x
        R[1, :] = local_y
        R[2, :] = local_z
        dof_i = np.array([node_i * 6 + j for j in range(6)])
        dof_j = np.array([node_j * 6 + j for j in range(6)])
        dof_elem = np.concatenate([dof_i, dof_j])
        u_elem = u_global[dof_elem]
        u_elem_local = np.zeros(12)
        for ii in range(2):
            u_elem_local[6*ii:6*ii+3] = R @ u_elem[6*ii:6*ii+3]
            u_elem_local[6*ii+3:6*ii+6] = R @ u_elem[6*ii+3:6*ii+6]
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        G = E / (2 * (1 + nu))
        Fx2 = E * A / L * (u_elem_local[6] - u_elem_local[0])
        Mx2 = G * J / L * (u_elem_local[9] - u_elem_local[3])
        My1 = 4 * E * I_y / L * u_elem_local[5] + 2 * E * I_y / L * u_elem_local[11]
        Mz1 = 4 * E * I_z / L * u_elem_local[4] + 2 * E * I_z / L * u_elem_local[10]
        My2 = 2 * E * I_y / L * u_elem_local[5] + 4 * E * I_y / L * u_elem_local[11]
        Mz2 = 2 * E * I_z / L * u_elem_local[4] + 4 * E * I_z / L * u_elem_local[10]
        kg_local = np.zeros((12, 12))
        kg_local[1, 1] = Fx2 / L
        kg_local[1, 5] = Fx2 / 6
        kg_local[1, 7] = -Fx2 / L
        kg_local[1, 11] = Fx2 / 6
        kg_local[5, 1] = Fx2 / 6
        kg_local[5, 5] = Fx2 * L / 30
        kg_local[5, 7] = -Fx2 / 6
        kg_local[5, 11] = -Fx2 * L / 30
        kg_local[2, 2] = Fx2 / L
        kg_local[2, 4] = -Fx2 / 6
        kg_local[2, 8] = -Fx2 / L
        kg_local[2, 10] = -Fx2 / 6
        kg_local[4, 2] = -Fx2 / 6
        kg_local[4, 4] = Fx2 * L / 30
        kg_local[4, 8] = Fx2 / 6
        kg_local[4, 10] = -Fx2 * L / 30
        kg_local[7, 1] = -Fx2 / L
        kg_local[7, 5] = -Fx2 / 6
        kg_local[7, 7] = Fx2 / L
        kg_local[7, 11] = -Fx2 / 6
        kg_local[8, 2] = -Fx2 / L
        kg_local[8, 4] = Fx2 / 6
        kg_local[8, 8] = Fx2 / L
        kg_local[8, 10] = Fx2 / 6
        kg_local[11, 1] = Fx2 / 6
        kg_local[11, 5] = -Fx2 * L / 30
        kg_local[11, 7] = -Fx2 / 6
        kg_local[11, 11] = Fx2 * L / 30
        kg_local[10, 2] = -Fx2 / 6
        kg_local[10, 4] = -Fx2 * L / 30
        kg_local[10, 8] = Fx2 / 6
        kg_local[10, 10] = Fx2 * L / 30
        kg_local[1, 5] += Mx2 / 6
        kg_local[5, 1] += Mx2 / 6
        kg_local[5, 5] += Mx2 * L / 30
        kg_local[1, 11] -= Mx2 / 6
        kg_local[11, 1] -= Mx2 / 6
        kg_local[5, 11] -= Mx2 * L / 30
        kg_local[11, 5] -= Mx2 * L / 30
        kg_local[11, 11] += Mx2 * L / 30
        kg_local[2, 4] -= Mx2 / 6
        kg_local[4, 2] -= Mx2 / 6
        kg_local[4, 4] += Mx2 * L / 30
        kg_local[2, 10] -= Mx2 / 6
        kg_local[10, 2] -= Mx2 / 6
        kg_local[4, 10] -= Mx2 * L / 30
        kg_local[10, 4] -= Mx2 * L / 30
        kg_local[10, 10] += Mx2 * L / 30
        kg_local[2, 4] += My1 / L * 0.1
        kg_local[1, 5] += Mz1 / L * 0.1
        kg_local = (kg_local + kg_local.T) / 2
        T = np.zeros((12, 12))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        kg_global = T.T @ kg_local @ T
        for ii, di in enumerate(dof_elem):
            for jj, dj in enumerate(dof_elem):
                K_g_global[di, dj] += kg_global[ii, jj]
    K_gg = K_global[np.ix_(free_dofs, free_dofs)]
    K_g_gg = K_g_global[np.ix_(free_dofs, free_dofs)]
    try:
        eigenvalues, eigenvectors = scipy.linalg.eigh(K_gg, -K_g_gg)
    except np.linalg.LinAlgError: