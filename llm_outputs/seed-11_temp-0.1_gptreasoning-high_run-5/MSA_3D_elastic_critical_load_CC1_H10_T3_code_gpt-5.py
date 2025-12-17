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
    try:
        coords = np.asarray(node_coords, dtype=float)
    except Exception as e:
        raise ValueError('node_coords must be convertible to a float ndarray') from e
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('node_coords must have shape (N, 3)')
    n_nodes = coords.shape[0]
    dof_per_node = 6
    ndof = n_nodes * dof_per_node

    def node_dofs(node_index: int):
        if not 0 <= node_index < n_nodes:
            raise ValueError(f'Element references node index {node_index}, but there are {n_nodes} nodes.')
        base = node_index * dof_per_node
        return np.arange(base, base + dof_per_node, dtype=int)

    def element_transform(ni: int, nj: int, local_z_hint: Optional[Sequence[float]]):
        xi = coords[ni]
        xj = coords[nj]
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element has zero or invalid length.')
        ex = dx / L
        if local_z_hint is None:
            gZ = np.array([0.0, 0.0, 1.0])
            gY = np.array([0.0, 1.0, 0.0])
            if np.linalg.norm(np.cross(ex, gZ)) < 1e-12:
                z_ref = gY.copy()
            else:
                z_ref = gZ.copy()
        else:
            z_arr = np.asarray(local_z_hint, dtype=float).reshape(3)
            normz = float(np.linalg.norm(z_arr))
            if normz <= 0.0 or not np.isfinite(normz):
                raise ValueError('Provided local_z must be a non-zero 3-vector.')
            z_ref = z_arr / normz
        y_temp = np.cross(z_ref, ex)
        ny = float(np.linalg.norm(y_temp))
        if ny < 1e-12:
            raise ValueError('Provided local_z is parallel to the element axis; pick a non-parallel vector.')
        ey = y_temp / ny
        ez = np.cross(ex, ey)
        R = np.column_stack((ex, ey, ez))
        T = np.zeros((12, 12), dtype=float)
        for blk in range(4):
            T[blk * 3:(blk + 1) * 3, blk * 3:(blk + 1) * 3] = R
        return (L, R, T)

    def element_stiffness_local(E: float, G: float, A: float, Iy: float, Iz: float, J: float, L: float):
        k = np.zeros((12, 12), dtype=float)
        ka = E * A / L
        k[0, 0] += ka
        k[0, 6] -= ka
        k[6, 0] -= ka
        k[6, 6] += ka
        kt = G * J / L
        k[3, 3] += kt
        k[3, 9] -= kt
        k[9, 3] -= kt
        k[9, 9] += kt
        c = E * Iz / L ** 3
        idx = [1, 5, 7, 11]
        kbz = np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2], [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2]], dtype=float) * c
        for a in range(4):
            for b in range(4):
                k[idx[a], idx[b]] += kbz[a, b]
        c2 = E * Iy / L ** 3
        idy = [2, 4, 8, 10]
        kby = np.array([[12, -6 * L, -12, -6 * L], [-6 * L, 4 * L ** 2, 6 * L, 2 * L ** 2], [-12, 6 * L, 12, 6 * L], [-6 * L, 2 * L ** 2, 6 * L, 4 * L ** 2]], dtype=float) * c2
        for a in range(4):
            for b in range(4):
                k[idy[a], idy[b]] += kby[a, b]
        return k

    def element_geometric_stiffness_local_from_N(N_axial: float, L: float):
        kg = np.zeros((12, 12), dtype=float)
        if L <= 0.0 or not np.isfinite(L):
            return kg
        coeff = N_axial / (30.0 * L)
        block = np.array([[36, 3 * L, -36, 3 * L], [3 * L, 4 * L ** 2, -3 * L, -1 * L ** 2], [-36, -3 * L, 36, -3 * L], [3 * L, -1 * L ** 2, -3 * L, 4 * L ** 2]], dtype=float) * coeff
        idx_z = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                kg[idx_z[a], idx_z[b]] += block[a, b]
        idx_y = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                kg[idx_y[a], idx_y[b]] += block[a, b]
        return kg
    K = np.zeros((ndof, ndof), dtype=float)
    element_data = []
    for e_idx, el in enumerate(elements):
        try:
            ni = int(el['node_i'])
            nj = int(el['node_j'])
            E = float(el['E'])
            nu = float(el['nu'])
            A = float(el['A'])
            Iy = float(el['I_y'])
            Iz = float(el['I_z'])
            J = float(el['J'])
        except Exception as e:
            raise ValueError(f'Element {e_idx} is missing required keys or has invalid values.') from e
        local_z_hint = el.get('local_z', None)
        L, R, T = element_transform(ni, nj, local_z_hint)
        if not np.isfinite(E) or not np.isfinite(nu):
            raise ValueError('Material properties must be finite.')
        if any((v <= 0.0 or not np.isfinite(v) for v in (A, Iy, Iz, J, L))):
            raise ValueError('Geometric properties and length must be positive and finite.')
        G = E / (2.0 * (1.0 + nu))
        Ke_local = element_stiffness_local(E, G, A, Iy, Iz, J, L)
        Ke_global = T @ Ke_local @ T.T
        dofs_i = node_dofs(ni)
        dofs_j = node_dofs(nj)
        edofs = np.concatenate((dofs_i, dofs_j))
        K[np.ix_(edofs, edofs)] += Ke_global
        element_data.append((ni, nj, edofs, T, Ke_local, L))
    P = np.zeros(ndof, dtype=float)
    for n, load in nodal_loads.items():
        try:
            load_arr = np.asarray(load, dtype=float).reshape(6)
        except Exception as e:
            raise ValueError(f'Invalid nodal load entry for node {n}.') from e
        if not 0 <= n < n_nodes:
            raise ValueError(f'Load specified for invalid node index {n}.')
        P[node_dofs(n)] += load_arr
    fixed = np.zeros(ndof, dtype=bool)
    for n, bc in boundary_conditions.items():
        if not 0 <= n < n_nodes:
            raise ValueError(f'Boundary condition specified for invalid node index {n}.')
        try:
            bc_arr = np.asarray(bc, dtype=int).reshape(6)
        except Exception as e:
            raise ValueError(f'Invalid boundary condition entry for node {n}.') from e
        if bc_arr.size != 6:
            raise ValueError(f'Boundary condition for node {n} must have 6 entries.')
        if np.any((bc_arr != 0) & (bc_arr != 1)):
            raise ValueError(f'Boundary condition entries must be 0 or 1 for node {n}.')
        fixed[node_dofs(n)] = bc_arr.astype(bool)
    free = ~fixed
    n_free = int(np.count_nonzero(free))
    if n_free == 0:
        raise ValueError('All DOFs are fixed; cannot perform analysis.')
    K_ff = K[np.ix_(free, free)]
    P_f = P[free]
    if not np.any(np.isfinite(P_f)):
        raise ValueError('Reference load vector contains invalid entries on free DOFs.')
    if np.linalg.norm(P_f) == 0.0:
        raise ValueError('Reference load vector is zero on free DOFs; cannot form geometric stiffness.')
    try:
        u_f = np.linalg.solve(K_ff, P_f)
    except np.linalg.LinAlgError as e:
        try:
            condK = np.linalg.cond(K_ff)
        except Exception:
            condK = np.inf
        raise ValueError(f'Reduced stiffness matrix is singular or ill-conditioned (cond={condK:.3e}).') from e
    u = np.zeros(ndof, dtype=float)
    u[free] = u_f
    Kg = np.zeros((ndof, ndof), dtype=float)
    for ni, nj, edofs, T, Ke_local, L in element_data:
        u_e_g = u[edofs]
        u_e_l = T.T @ u_e_g
        f_e_l = Ke_local @ u_e_l
        N_i = float(f_e_l[0])
        N_j = float(-f_e_l[6])
        N_axial = 0.5 * (N_i + N_j)
        Kg_local = element_geometric_stiffness_local_from_N(N_axial, L)
        Kg_global = T @ Kg_local @ T.T
        Kg[np.ix_(edofs, edofs)] += Kg_global
    Kg_ff = Kg[np.ix_(free, free)]
    if not np.isfinite(Kg_ff).all() or np.linalg.norm(Kg_ff, ord='fro') < 1e-16:
        raise ValueError('Geometric stiffness matrix is zero or invalid; check reference load and model.')
    try:
        eigvals, eigvecs = scipy.linalg.eig(K_ff, -Kg_ff)
    except Exception as e:
        raise ValueError('Generalized eigenvalue problem failed to solve.') from e
    eigvals = np.asarray(eigvals)
    tol_imag = 1e-08
    real_mask = np.abs(eigvals.imag) <= tol_imag * (1.0 + np.abs(eigvals.real))
    pos_mask = eigvals.real > 1e-12
    valid_mask = real_mask & pos_mask & np.isfinite(eigvals.real)
    if not np.any(valid_mask):
        raise ValueError('No positive real eigenvalue found for the buckling problem.')
    valid_vals = eigvals.real[valid_mask]
    valid_vecs = eigvecs[:, valid_mask]
    idx_min = int(np.argmin(valid_vals))
    lam = float(valid_vals[idx_min])
    phi_free = valid_vecs[:, idx_min]
    if np.linalg.norm(phi_free.imag) > 1e-06 * (1.0 + np.linalg.norm(phi_free.real)):
        raise ValueError('Selected eigenvector has significant complex part.')
    phi_free = phi_free.real
    phi = np.zeros(ndof, dtype=float)
    phi[free] = phi_free
    return (lam, phi)