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
    n_dofs = 6 * n_nodes

    def get_transformation_matrix(node_i_coords, node_j_coords, local_z=None):
        """Compute the 12x12 transformation matrix from local to global coordinates."""
        dx = node_j_coords - node_i_coords
        L = np.linalg.norm(dx)
        x_axis = dx / L
        if local_z is None:
            if abs(x_axis[2]) > 0.9:
                local_z = np.array([0.0, 1.0, 0.0])
            else:
                local_z = np.array([0.0, 0.0, 1.0])
        else:
            local_z = np.array(local_z, dtype=float)
            local_z = local_z / np.linalg.norm(local_z)
        y_axis = np.cross(local_z, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        R = np.array([x_axis, y_axis, z_axis])
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        return (T, L, R)

    def get_local_elastic_stiffness(E, nu, A, I_y, I_z, J, L):
        """Compute the 12x12 local elastic stiffness matrix for a 3D beam element."""
        G = E / (2 * (1 + nu))
        k = np.zeros((12, 12))
        k[0, 0] = E * A / L
        k[0, 6] = -E * A / L
        k[6, 0] = -E * A / L
        k[6, 6] = E * A / L
        k[3, 3] = G * J / L
        k[3, 9] = -G * J / L
        k[9, 3] = -G * J / L
        k[9, 9] = G * J / L
        k[1, 1] = 12 * E * I_z / L ** 3
        k[1, 5] = 6 * E * I_z / L ** 2
        k[1, 7] = -12 * E * I_z / L ** 3
        k[1, 11] = 6 * E * I_z / L ** 2
        k[5, 1] = 6 * E * I_z / L ** 2
        k[5, 5] = 4 * E * I_z / L
        k[5, 7] = -6 * E * I_z / L ** 2
        k[5, 11] = 2 * E * I_z / L
        k[7, 1] = -12 * E * I_z / L ** 3
        k[7, 5] = -6 * E * I_z / L ** 2
        k[7, 7] = 12 * E * I_z / L ** 3
        k[7, 11] = -6 * E * I_z / L ** 2
        k[11, 1] = 6 * E * I_z / L ** 2
        k[11, 5] = 2 * E * I_z / L
        k[11, 7] = -6 * E * I_z / L ** 2
        k[11, 11] = 4 * E * I_z / L
        k[2, 2] = 12 * E * I_y / L ** 3
        k[2, 4] = -6 * E * I_y / L ** 2
        k[2, 8] = -12 * E * I_y / L ** 3
        k[2, 10] = -6 * E * I_y / L ** 2
        k[4, 2] = -6 * E * I_y / L ** 2
        k[4, 4] = 4 * E * I_y / L
        k[4, 8] = 6 * E * I_y / L ** 2
        k[4, 10] = 2 * E * I_y / L
        k[8, 2] = -12 * E * I_y / L ** 3
        k[8, 4] = 6 * E * I_y / L ** 2
        k[8, 8] = 12 * E * I_y / L ** 3
        k[8, 10] = 6 * E * I_y / L ** 2
        k[10, 2] = -6 * E * I_y / L ** 2
        k[10, 4] = 2 * E * I_y / L
        k[10, 8] = 6 * E * I_y / L ** 2
        k[10, 10] = 4 * E * I_y / L
        return k

    def get_local_geometric_stiffness(L, Fx2, Mx2, My1, Mz1, My2, Mz2):
        """Compute the 12x12 local geometric stiffness matrix for a 3D beam element."""
        kg = np.zeros((12, 12))
        P = -Fx2
        a = P / L
        b = P / 10
        c = P * L / 30
        kg[1, 1] = 6 * a / 5
        kg[1, 5] = a * L / 10
        kg[1, 7] = -6 * a / 5
        kg[1, 11] = a * L / 10
        kg[5, 1] = a * L / 10
        kg[5, 5] = 2 * a * L ** 2 / 15
        kg[5, 7] = -a * L / 10
        kg[5, 11] = -a * L ** 2 / 30
        kg[7, 1] = -6 * a / 5
        kg[7, 5] = -a * L / 10
        kg[7, 7] = 6 * a / 5
        kg[7, 11] = -a * L / 10
        kg[11, 1] = a * L / 10
        kg[11, 5] = -a * L ** 2 / 30
        kg[11, 7] = -a * L / 10
        kg[11, 11] = 2 * a * L ** 2 / 15
        kg[2, 2] = 6 * a / 5
        kg[2, 4] = -a * L / 10
        kg[2, 8] = -6 * a / 5
        kg[2, 10] = -a * L / 10
        kg[4, 2] = -a * L / 10
        kg[4, 4] = 2 * a * L ** 2 / 15
        kg[4, 8] = a * L / 10
        kg[4, 10] = -a * L ** 2 / 30
        kg[8, 2] = -6 * a / 5
        kg[8, 4] = a * L / 10
        kg[8, 8] = 6 * a / 5
        kg[8, 10] = a * L / 10
        kg[10, 2] = -a * L / 10
        kg[10, 4] = -a * L ** 2 / 30
        kg[10, 8] = a * L / 10
        kg[10, 10] = 2 * a * L ** 2 / 15
        if abs(Mx2) > 1e-14:
            m = Mx2 / L
            kg[1, 3] = m
            kg[3, 1] = m
            kg[2, 3] = -m
            kg[3, 2] = -m
            kg[7, 3] = -m
            kg[3, 7] = -m
            kg[8, 3] = m
            kg[3, 8] = m
            kg[1, 9] = -m
            kg[9, 1] = -m
            kg[2, 9] = m
            kg[9, 2] = m
            kg[7, 9] = m
            kg[9, 7] = m
            kg[8, 9] = -m
            kg[9, 8] = -m
        return kg

    def get_element_internal_forces(elem, node_coords, u_global):
        """Extract element internal forces from global displacement vector."""
        node_i = elem['node_i']
        node_j = elem['node_j']
        node_i_coords = node_coords[node_i]
        node_j_coords = node_coords[node_j]
        local_z = elem.get('local_z', None)
        (T, L, R) = get_transformation_matrix(node_i_coords, node_j_coords, local_z)
        dof_i = list(range(6 * node_i, 6 * node_i + 6))
        dof_j = list(range(6 * node_j, 6 * node_j + 6))
        dofs = dof_i + dof_j
        u_elem_global = u_global[dofs]
        u_elem_local = T @ u_elem_global
        k_local = get_local_elastic_stiffness(elem['E'], elem['nu'], elem['A'], elem['I_y'], elem['I_z'], elem['J'], L)
        f_local = k_local @ u_elem_local
        return (f_local, L)
    K = np.zeros((n_dofs, n_dofs))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        node_i_coords = node_coords[node_i]
        node_j_coords = node_coords[node_j]
        local_z = elem.get('local_z', None)
        (T, L, R) = get_transformation_matrix(node_i_coords, node_j_coords, local_z)
        k_local = get_local_elastic_stiffness(elem['E'], elem['nu'], elem['A'], elem['I_y'], elem['I_z'], elem['J'], L)
        k_global = T.T @ k_local @ T
        dof_i = list(range(6 * node_i, 6 * node_i + 6))
        dof_j = list(range(6 * node_j, 6 * node_j + 6))
        dofs = dof_i + dof_j
        for (ii, di) in enumerate(dofs):
            for (jj, dj) in enumerate(dofs):
                K[di, dj] += k_global[ii, jj]
    P = np.zeros(n_dofs)
    for (node_idx, loads) in nodal_loads.items():
        for (i, load) in enumerate(loads):
            P[6 * node_idx + i] = load
    fixed_dofs = set()
    for (node_idx, bc) in boundary_conditions.items():
        for (i, is_fixed) in enumerate(bc):
            if is_fixed:
                fixed_dofs.add(6 * node_idx + i)
    all_dofs = set(range(n_dofs))
    free_dofs = sorted(all_dofs - fixed_dofs)
    free_dofs = np.array(free_dofs)
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    u_f = np.linalg.solve(K_ff, P_f)
    u_global = np.zeros(n_dofs)
    u_global[free_dofs] = u_f
    K_g = np.zeros((n_dofs, n_dofs))
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        node_i_coords = node_coords[node_i]
        node_j_coords = node_coords[node_j]
        local_z = elem.get('local_z', None)
        (T, L, R) = get_transformation_matrix(node_i_coords, node_j_coords, local_z)
        (f_local, L) = get_element_internal_forces(elem, node_coords, u_global)
        Fx2 = f_local[6]
        Mx2 = f_local[9]
        My1 = f_local[4]
        Mz1 = f_local[5]
        My2 = f_local[10]
        Mz2 = f_local[11]
        kg_local = get_local_geometric_stiffness(L, Fx2, Mx2, My1, Mz1, My2, Mz2)
        kg_global = T.T @ kg_local @ T
        dof_i = list(range(6 * node_i, 6 * node_i + 6))
        dof_j = list(range(6 * node_j, 6 * node_j + 6))
        dofs = dof_i + dof_j
        for (ii, di) in enumerate(dofs):
            for (jj, dj) in enumerate(dofs):
                K_g[di, dj] += kg_global[ii, jj]
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    Kg_ff = K_g[np.ix_(free_dofs, free_dofs)]
    (eigenvalues, eigenvectors) = scipy.linalg.eig(K_ff, -Kg_ff)
    tol = 1e-10
    valid_indices = []
    valid_eigenvalues = []
    for (i, ev) in enumerate(eigenvalues):
        if abs(ev.imag) < tol * max(abs(ev.real), 1.0):
            if ev.real > tol:
                valid_indices.append(i)
                valid_eigenvalues.append(ev.real)
    if len(valid_indices) == 0:
        raise ValueError('No positive eigenvalue found')
    min_idx = valid_indices[np.argmin(valid_eigenvalues)]
    lambda_cr = eigenvalues[min_idx].real
    phi_f = eigenvectors[:, min_idx].real
    phi_global = np.zeros(n_dofs)
    phi_global[free_dofs] = phi_f
    return (float(lambda_cr), phi_global)