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
    import numpy as np
    import scipy
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an (N,3) array of floats.')
    n_nodes = node_coords.shape[0]
    dof_per_node = 6
    n_dof = n_nodes * dof_per_node

    def _node_dofs(n):
        start = n * dof_per_node
        return np.arange(start, start + dof_per_node, dtype=int)

    def _rotation_matrix_and_length(ni, nj, local_z_vec):
        xi = node_coords[ni]
        xj = node_coords[nj]
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not L > 0.0:
            raise ValueError('Element length must be positive (distinct node coordinates).')
        ex = dx / L
        if local_z_vec is None:
            zc = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(ex, zc)) > 0.999:
                zc = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            zc = np.asarray(local_z_vec, dtype=float).reshape(3)
            nz = np.linalg.norm(zc)
            if not nz > 0.0:
                raise ValueError('Provided local_z vector must be non-zero.')
            zc = zc / nz
        y_temp = np.cross(zc, ex)
        ny = np.linalg.norm(y_temp)
        if ny < 1e-12:
            alt = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(ex, alt)) > 0.999:
                alt = np.array([0.0, 1.0, 0.0], dtype=float)
            y_temp = np.cross(alt, ex)
            ny = np.linalg.norm(y_temp)
            if ny < 1e-12:
                raise ValueError('Failed to construct a valid local coordinate system.')
        ey = y_temp / ny
        ez = np.cross(ex, ey)
        nz = np.linalg.norm(ez)
        if not nz > 0.0:
            raise ValueError('Failed to construct a valid orthonormal basis.')
        ez = ez / nz
        R = np.column_stack((ex, ey, ez))
        return (R, L)

    def _element_stiffness_local(E, G, A, Iy, Iz, J, L):
        k = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        k[0, 0] += k_ax
        k[0, 6] -= k_ax
        k[6, 0] -= k_ax
        k[6, 6] += k_ax
        k_t = G * J / L
        k[3, 3] += k_t
        k[3, 9] -= k_t
        k[9, 3] -= k_t
        k[9, 9] += k_t
        L2 = L * L
        L3 = L2 * L
        c = E * Iz / L3
        kbz = c * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float)
        idx_y = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                k[idx_y[a], idx_y[b]] += kbz[a, b]
        c2 = E * Iy / L3
        kby = c2 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float)
        idx_z = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                k[idx_z[a], idx_z[b]] += kby[a, b]
        return k

    def _element_geometric_stiffness_local(N_axial, L):
        kg = np.zeros((12, 12), dtype=float)
        if L <= 0.0:
            return kg
        coef = N_axial / (30.0 * L)
        L2 = L * L
        base = np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L2, -3.0 * L, -1.0 * L2], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -1.0 * L2, -3.0 * L, 4.0 * L2]], dtype=float)
        kg_plane = coef * base
        idx_y = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                kg[idx_y[a], idx_y[b]] += kg_plane[a, b]
        idx_z = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                kg[idx_z[a], idx_z[b]] += kg_plane[a, b]
        return kg

    def _transformation_matrix(R):
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        return T
    K = np.zeros((n_dof, n_dof), dtype=float)
    P = np.zeros(n_dof, dtype=float)
    if nodal_loads is not None:
        for (n, load) in nodal_loads.items():
            if n < 0 or n >= n_nodes:
                raise ValueError('nodal_loads contains an invalid node index.')
            load = np.asarray(load, dtype=float).reshape(-1)
            if load.size != 6:
                raise ValueError('Each nodal load must be a 6-element sequence.')
            P[_node_dofs(n)] += load
    if elements is None:
        elements = []
    for e in elements:
        if not all((k in e for k in ('node_i', 'node_j', 'E', 'nu', 'A', 'I_y', 'I_z', 'J'))):
            raise ValueError("Each element dict must include 'node_i','node_j','E','nu','A','I_y','I_z','J'.")
        ni = int(e['node_i'])
        nj = int(e['node_j'])
        if ni < 0 or nj < 0 or ni >= n_nodes or (nj >= n_nodes):
            raise ValueError('Element node indices out of range.')
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['I_y'])
        Iz = float(e['I_z'])
        J = float(e['J'])
        local_z = e.get('local_z', None)
        (R, L) = _rotation_matrix_and_length(ni, nj, local_z)
        G = E / (2.0 * (1.0 + nu))
        Ke_local = _element_stiffness_local(E, G, A, Iy, Iz, J, L)
        T = _transformation_matrix(R)
        Ke_global = T @ Ke_local @ T.T
        edofs = np.concatenate((_node_dofs(ni), _node_dofs(nj)))
        np.add.at(K, (edofs[:, None], edofs[None, :]), Ke_global)
    fixed = np.zeros(n_dof, dtype=bool)
    if boundary_conditions is not None:
        for (n, bc) in boundary_conditions.items():
            if n < 0 or n >= n_nodes:
                raise ValueError('boundary_conditions contains an invalid node index.')
            bc_arr = np.asarray(bc, dtype=int).reshape(-1)
            if bc_arr.size != 6:
                raise ValueError('Each boundary condition must be a 6-element sequence of 0/1.')
            bc_bool = bc_arr.astype(bool)
            ndofs = _node_dofs(n)
            fixed[ndofs] = bc_bool
    free = ~fixed
    free_idx = np.where(free)[0]
    if free_idx.size == 0:
        raise ValueError('All DOFs are constrained; no free DOFs to solve.')
    K_ff = K[np.ix_(free, free)]
    P_f = P[free]
    try:
        u_f = np.linalg.solve(K_ff, P_f)
    except np.linalg.LinAlgError as err:
        raise ValueError(f'Linear static solve failed (possibly singular K_ff): {err}')
    try:
        cond_Kff = np.linalg.cond(K_ff)
        if not np.isfinite(cond_Kff) or cond_Kff > 100000000000000.0:
            raise ValueError('Reduced stiffness matrix K_ff is ill-conditioned.')
    except Exception:
        pass
    u = np.zeros(n_dof, dtype=float)
    u[free] = u_f
    Kg = np.zeros((n_dof, n_dof), dtype=float)
    for e in elements:
        ni = int(e['node_i'])
        nj = int(e['node_j'])
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['I_y'])
        Iz = float(e['I_z'])
        J = float(e['J'])
        local_z = e.get('local_z', None)
        (R, L) = _rotation_matrix_and_length(ni, nj, local_z)
        _ = E / (2.0 * (1.0 + nu))
        T = _transformation_matrix(R)
        edofs = np.concatenate((_node_dofs(ni), _node_dofs(nj)))
        d_e_global = u[edofs]
        d_e_local = T.T @ d_e_global
        N_axial = E * A / L * (d_e_local[6] - d_e_local[0])
        Kg_local = _element_geometric_stiffness_local(N_axial, L)
        Kg_global = T @ Kg_local @ T.T
        np.add.at(Kg, (edofs[:, None], edofs[None, :]), Kg_global)
    Kg_ff = Kg[np.ix_(free, free)]
    norm_Kg = np.linalg.norm(Kg_ff, ord='fro')
    if not norm_Kg > 0.0:
        raise ValueError('Geometric stiffness on free DOFs is zero; cannot perform buckling analysis.')
    try:
        (w, v) = scipy.linalg.eig(K_ff, -Kg_ff)
    except Exception as err:
        raise ValueError(f'Generalized eigenvalue solve failed: {err}')
    w = np.asarray(w)
    v = np.asarray(v)
    w_real = w.real
    w_imag = w.imag
    imag_tol = 1e-06
    real_tol = 1e-09
    mask_real = np.abs(w_imag) <= imag_tol * (1.0 + np.abs(w_real))
    mask_pos = w_real > real_tol
    mask = mask_real & mask_pos
    if not np.any(mask):
        raise ValueError('No positive real eigenvalue found for buckling problem.')
    idxs = np.where(mask)[0]
    sel = idxs[np.argmin(w_real[idxs])]
    lam = float(w_real[sel])
    phi_f = v[:, sel].real
    phi = np.zeros(n_dof, dtype=float)
    phi[free] = phi_f
    return (lam, phi)