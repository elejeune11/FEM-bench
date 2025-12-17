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
    import numpy as _np
    import scipy.linalg as _sla

    def _safe_array(x, shape=None):
        a = _np.asarray(x, dtype=float)
        if shape is not None and a.shape != shape:
            raise ValueError('Array has unexpected shape.')
        return a
    node_coords = _safe_array(node_coords)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be of shape (N, 3)')
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    K = _np.zeros((ndof, ndof), dtype=float)
    Kg = _np.zeros((ndof, ndof), dtype=float)
    P = _np.zeros(ndof, dtype=float)
    for n_idx, load in (nodal_loads or {}).items():
        if n_idx < 0 or n_idx >= n_nodes:
            raise ValueError('nodal_loads contains invalid node index.')
        l = _np.asarray(load, dtype=float).reshape(-1)
        if l.size != 6:
            raise ValueError('Each nodal load must have 6 components.')
        dofs = slice(6 * n_idx, 6 * n_idx + 6)
        P[dofs] += l

    def _element_rotation_and_length(ni, nj, local_z_spec):
        xi = node_coords[ni]
        xj = node_coords[nj]
        dx = xj - xi
        L = _np.linalg.norm(dx)
        if not _np.isfinite(L) or L <= 0.0:
            raise ValueError('Element has zero or invalid length.')
        ex = dx / L
        if local_z_spec is None:
            ref = _np.array([0.0, 0.0, 1.0])
            if abs(_np.dot(ex, ref)) > 1.0 - 1e-08:
                ref = _np.array([0.0, 1.0, 0.0])
        else:
            ref = _np.asarray(local_z_spec, dtype=float).reshape(3)
            nr = _np.linalg.norm(ref)
            if nr == 0:
                raise ValueError('local_z must be non-zero.')
            ref = ref / nr
        if abs(_np.dot(ref, ex)) > 1.0 - 1e-08:
            alt = _np.array([0.0, 1.0, 0.0]) if abs(ex[1]) < 0.9 else _np.array([1.0, 0.0, 0.0])
            ref = alt - _np.dot(alt, ex) * ex
            nr = _np.linalg.norm(ref)
            if nr < 1e-12:
                alt = _np.array([1.0, 0.0, 0.0])
                ref = alt - _np.dot(alt, ex) * ex
                nr = _np.linalg.norm(ref)
                if nr < 1e-12:
                    raise ValueError('Cannot determine a valid local z-axis.')
            ref = ref / nr
        ey = ref - _np.dot(ref, ex) * ex
        n_ey = _np.linalg.norm(ey)
        if n_ey < 1e-12:
            alt = _np.array([0.0, 1.0, 0.0]) if abs(ex[1]) < 0.9 else _np.array([1.0, 0.0, 0.0])
            ey = alt - _np.dot(alt, ex) * ex
            n_ey = _np.linalg.norm(ey)
            if n_ey < 1e-12:
                raise ValueError('Failed to construct local axes.')
        ey = ey / n_ey
        ez = _np.cross(ex, ey)
        n_ez = _np.linalg.norm(ez)
        if n_ez < 1e-12:
            raise ValueError('Failed to construct orthonormal triad.')
        ez = ez / n_ez
        R = _np.column_stack((ex, ey, ez))
        return (R, L)

    def _element_stiffness_local(E, G, A, Iy, Iz, J, L):
        k = _np.zeros((12, 12), dtype=float)
        EA_L = E * A / L
        GJ_L = G * J / L
        EIy = E * Iy
        EIz = E * Iz
        L2 = L * L
        L3 = L2 * L
        k[0, 0] += EA_L
        k[0, 6] += -EA_L
        k[6, 0] += -EA_L
        k[6, 6] += EA_L
        k[3, 3] += GJ_L
        k[3, 9] += -GJ_L
        k[9, 3] += -GJ_L
        k[9, 9] += GJ_L
        c1 = 12.0 * EIz / L3
        c2 = 6.0 * EIz / L2
        c3 = 4.0 * EIz / L
        c4 = 2.0 * EIz / L
        idx_v_i, idx_rz_i, idx_v_j, idx_rz_j = (1, 5, 7, 11)
        k[idx_v_i, idx_v_i] += c1
        k[idx_v_i, idx_rz_i] += c2
        k[idx_v_i, idx_v_j] += -c1
        k[idx_v_i, idx_rz_j] += c2
        k[idx_rz_i, idx_v_i] += c2
        k[idx_rz_i, idx_rz_i] += c3
        k[idx_rz_i, idx_v_j] += -c2
        k[idx_rz_i, idx_rz_j] += c4
        k[idx_v_j, idx_v_i] += -c1
        k[idx_v_j, idx_rz_i] += -c2
        k[idx_v_j, idx_v_j] += c1
        k[idx_v_j, idx_rz_j] += -c2
        k[idx_rz_j, idx_v_i] += c2
        k[idx_rz_j, idx_rz_i] += c4
        k[idx_rz_j, idx_v_j] += -c2
        k[idx_rz_j, idx_rz_j] += c3
        c1 = 12.0 * EIy / L3
        c2 = 6.0 * EIy / L2
        c3 = 4.0 * EIy / L
        c4 = 2.0 * EIy / L
        idx_w_i, idx_ry_i, idx_w_j, idx_ry_j = (2, 4, 8, 10)
        k[idx_w_i, idx_w_i] += c1
        k[idx_w_i, idx_ry_i] += -c2
        k[idx_w_i, idx_w_j] += -c1
        k[idx_w_i, idx_ry_j] += -c2
        k[idx_ry_i, idx_w_i] += -c2
        k[idx_ry_i, idx_ry_i] += c3
        k[idx_ry_i, idx_w_j] += c2
        k[idx_ry_i, idx_ry_j] += c4
        k[idx_w_j, idx_w_i] += -c1
        k[idx_w_j, idx_ry_i] += c2
        k[idx_w_j, idx_w_j] += c1
        k[idx_w_j, idx_ry_j] += c2
        k[idx_ry_j, idx_w_i] += -c2
        k[idx_ry_j, idx_ry_i] += c4
        k[idx_ry_j, idx_w_j] += c2
        k[idx_ry_j, idx_ry_j] += c3
        return k

    def _element_geometric_stiffness_local(N_axial, L):
        kg = _np.zeros((12, 12), dtype=float)
        if abs(L) <= 0.0:
            return kg
        f = N_axial / (30.0 * L)
        L1 = L
        L2 = L * L
        K2d = _np.array([[36.0, 3.0 * L1, -36.0, 3.0 * L1], [3.0 * L1, 4.0 * L2, -3.0 * L1, -1.0 * L2], [-36.0, -3.0 * L1, 36.0, -3.0 * L1], [3.0 * L1, -1.0 * L2, -3.0 * L1, 4.0 * L2]], dtype=float) * f
        idx = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                kg[idx[a], idx[b]] += K2d[a, b]
        idx = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                kg[idx[a], idx[b]] += K2d[a, b]
        return kg

    def _T_gl_from_R(R):
        T = _np.zeros((12, 12), dtype=float)
        for n in (0, 6):
            T[n:n + 3, n:n + 3] = R
            T[n + 3:n + 6, n + 3:n + 6] = R
        return T

    def _element_dofs(ni, nj):
        return _np.r_[6 * ni + _np.arange(6), 6 * nj + _np.arange(6)]
    element_cache = []
    for e in elements:
        ni = int(e['node_i'])
        nj = int(e['node_j'])
        if not (0 <= ni < n_nodes and 0 <= nj < n_nodes):
            raise ValueError('Element references invalid node indices.')
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['I_y'])
        Iz = float(e['I_z'])
        J = float(e['J'])
        local_z_spec = e.get('local_z', None)
        R, L = _element_rotation_and_length(ni, nj, local_z_spec)
        G = E / (2.0 * (1.0 + nu))
        k_loc = _element_stiffness_local(E, G, A, Iy, Iz, J, L)
        T_gl = _T_gl_from_R(R)
        k_glob = T_gl @ k_loc @ T_gl.T
        dofs = _element_dofs(ni, nj)
        K[_np.ix_(dofs, dofs)] += k_glob
        element_cache.append((ni, nj, E, G, A, Iy, Iz, J, R, L, T_gl))
    fixed = _np.zeros(ndof, dtype=bool)
    for n_idx, bc in (boundary_conditions or {}).items():
        if n_idx < 0 or n_idx >= n_nodes:
            raise ValueError('boundary_conditions contains invalid node index.')
        vals = _np.asarray(bc, dtype=int).reshape(-1)
        if vals.size != 6:
            raise ValueError('Each boundary_conditions entry must have 6 values.')
        for k in range(6):
            if vals[k]:
                fixed[6 * n_idx + k] = True
    free = ~fixed
    n_free = int(_np.sum(free))
    if n_free <= 0:
        raise ValueError('All DOFs are fixed; no free DOFs remain.')
    K_ff = 0.5 * (K[_np.ix_(free, free)] + K[_np.ix_(free, free)].T)
    try:
        _np.linalg.cholesky(K_ff)
    except _np.linalg.LinAlgError as exc:
        raise ValueError('Reduced stiffness matrix is not positive definite. Check boundary conditions.') from exc
    u = _np.zeros(ndof, dtype=float)
    P_f = P[free]
    try:
        u[free] = _np.linalg.solve(K_ff, P_f)
    except _np.linalg.LinAlgError as exc:
        raise ValueError('Linear solve failed; K_ff may be singular or ill-conditioned.') from exc
    Kg.fill(0.0)
    for ni, nj, E, G, A, Iy, Iz, J, R, L, T_gl in element_cache:
        dofs = _element_dofs(ni, nj)
        u_e_g = u[dofs]
        u_e_l = T_gl.T @ u_e_g
        du_ax = u_e_l[6 + 0] - u_e_l[0]
        N_axial = E * A / L * du_ax
        kg_loc = _element_geometric_stiffness_local(N_axial, L)
        kg_glob = T_gl @ kg_loc @ T_gl.T
        Kg[_np.ix_(dofs, dofs)] += kg_glob
    Kg_ff = 0.5 * (Kg[_np.ix_(free, free)] + Kg[_np.ix_(free, free)].T)
    norm_Kg = _np.linalg.norm(Kg_ff, ord='fro')
    if not _np.isfinite(norm_Kg) or norm_Kg <= 1e-14:
        raise ValueError('Geometric stiffness matrix is zero or invalid; no buckling prediction possible for the provided loads.')
    try:
        w, V = _sla.eig(K_ff, -Kg_ff)
    except Exception as exc:
        raise ValueError('Eigenvalue computation failed.') from exc
    w = _np.asarray(w)
    V = _np.asarray(V)
    is_finite = _np.isfinite(w.real) & _np.isfinite(w.imag)
    w = w[is_finite]
    V = V[:, is_finite]
    if w.size == 0:
        raise ValueError('No finite eigenvalues returned.')
    tol_imag = 1e-07
    real_mask = _np.abs(w.imag) <= tol_imag * _np.maximum(1.0, _np.abs(w.real))
    w_real = w.real[real_mask]
    V_real = V[:, real_mask]
    pos_mask = w_real > 1e-12
    w_pos = w_real[pos_mask]
    V_pos = V_real[:, pos_mask]
    if w_pos.size == 0:
        raise ValueError('No positive real eigenvalue found.')
    idx_min = int(_np.argmin(w_pos))
    lam = float(w_pos[idx_min])
    phi_f = V_pos[:, idx_min].real
    phi = _np.zeros(ndof, dtype=float)
    phi[free] = phi_f
    return (lam, phi)