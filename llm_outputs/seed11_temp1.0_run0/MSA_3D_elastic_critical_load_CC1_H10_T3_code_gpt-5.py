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
    from numpy.linalg import norm as _norm
    tol_len = 1e-12
    tol_parallel = 1e-08
    tol_solve = 1e-12
    tol_imag = 1e-08
    tol_pos = 1e-12
    node_coords = _np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an (N,3) array of node coordinates.')
    n_nodes = node_coords.shape[0]
    dof_per_node = 6
    n_dof = n_nodes * dof_per_node
    bc_flags = _np.zeros((n_nodes, dof_per_node), dtype=int)
    if boundary_conditions is not None:
        for (ni, flags) in boundary_conditions.items():
            if ni < 0 or ni >= n_nodes:
                raise ValueError('Boundary condition node index out of range.')
            flags_arr = _np.asarray(list(flags), dtype=int).reshape(-1)
            if flags_arr.size != 6:
                raise ValueError('Each boundary condition entry must have 6 values.')
            bc_flags[ni, :] = flags_arr
    fixed_mask = bc_flags.reshape(-1).astype(bool)
    free_mask = ~fixed_mask
    free_dofs = _np.nonzero(free_mask)[0]
    fixed_dofs = _np.nonzero(fixed_mask)[0]
    if free_dofs.size == 0:
        raise ValueError('All DOFs are fixed; no solution space remains.')
    P = _np.zeros(n_dof, dtype=float)
    if nodal_loads is not None:
        for (ni, loads) in nodal_loads.items():
            if ni < 0 or ni >= n_nodes:
                raise ValueError('Load node index out of range.')
            loads_arr = _np.asarray(list(loads), dtype=float).reshape(-1)
            if loads_arr.size != 6:
                raise ValueError('Each nodal load entry must have 6 values.')
            P[ni * dof_per_node:(ni + 1) * dof_per_node] = loads_arr

    def _element_rotation(ri: _np.ndarray, rj: _np.ndarray, local_z_pref: _np.ndarray | None):
        x_vec = rj - ri
        L = _norm(x_vec)
        if not _np.isfinite(L) or L <= tol_len:
            raise ValueError('Element has zero or invalid length.')
        ex = x_vec / L
        if local_z_pref is None:
            z_pref = _np.array([0.0, 0.0, 1.0])
            if abs(ex @ z_pref) > 1.0 - 1e-06:
                z_pref = _np.array([0.0, 1.0, 0.0])
        else:
            z_pref = _np.asarray(local_z_pref, dtype=float).reshape(3)
            if _norm(z_pref) <= tol_len:
                raise ValueError('Provided local_z has near-zero norm.')
        y_temp = _np.cross(z_pref, ex)
        if _norm(y_temp) <= tol_parallel:
            alt = _np.array([0.0, 1.0, 0.0])
            if abs(ex @ alt) > 1.0 - 1e-06:
                alt = _np.array([1.0, 0.0, 0.0])
            y_temp = _np.cross(alt, ex)
            if _norm(y_temp) <= tol_parallel:
                raise ValueError('Cannot determine a valid local axis system (local_z parallel to element).')
        ey = y_temp / _norm(y_temp)
        ez = _np.cross(ex, ey)
        ez = ez / _norm(ez)
        R = _np.column_stack((ex, ey, ez))
        return (R, L)

    def _T_from_R(R: _np.ndarray):
        T = _np.zeros((12, 12), dtype=float)
        for k in range(4):
            T[k * 3:(k + 1) * 3, k * 3:(k + 1) * 3] = R
        return T

    def _add_bending(k: _np.ndarray, EI: float, L: float, vi: int, ri: int, vj: int, rj: int):
        L2 = L * L
        L3 = L2 * L
        c = EI / L3
        m = _np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float) * c
        idx = [vi, ri, vj, rj]
        for a in range(4):
            for b in range(4):
                k[idx[a], idx[b]] += m[a, b]

    def _kg_bending_block(N: float, L: float):
        c = N / (30.0 * L)
        return c * _np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L * L, -3.0 * L, -1.0 * L * L], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -1.0 * L * L, -3.0 * L, 4.0 * L * L]], dtype=float)
    K = _np.zeros((n_dof, n_dof), dtype=float)
    Kg = _np.zeros((n_dof, n_dof), dtype=float)
    if elements is None:
        raise ValueError('elements must be a non-empty iterable of element dictionaries.')
    elem_data = []
    for e in elements:
        try:
            ni = int(e['node_i'])
            nj = int(e['node_j'])
            E = float(e['E'])
            nu = float(e['nu'])
            A = float(e['A'])
            Iy = float(e['I_y'])
            Iz = float(e['I_z'])
            J = float(e['J'])
        except Exception as ex:
            raise ValueError(f'Element has missing/invalid properties: {ex}')
        if not (0 <= ni < n_nodes and 0 <= nj < n_nodes):
            raise ValueError('Element node indices out of range.')
        ri = node_coords[ni]
        rj = node_coords[nj]
        local_z_pref = e.get('local_z', None)
        (R, L) = _element_rotation(ri, rj, None if local_z_pref is None else _np.asarray(local_z_pref, dtype=float))
        G = E / (2.0 * (1.0 + nu))
        k = _np.zeros((12, 12), dtype=float)
        EA_L = E * A / L
        k[0, 0] += EA_L
        k[0, 6] -= EA_L
        k[6, 0] -= EA_L
        k[6, 6] += EA_L
        GJ_L = G * J / L
        k[3, 3] += GJ_L
        k[3, 9] -= GJ_L
        k[9, 3] -= GJ_L
        k[9, 9] += GJ_L
        _add_bending(k, E * Iz, L, 1, 5, 7, 11)
        _add_bending(k, E * Iy, L, 2, 4, 8, 10)
        T = _T_from_R(R)
        kg = T @ k @ T.T
        edofs = _np.r_[ni * 6 + _np.arange(6), nj * 6 + _np.arange(6)]
        for a in range(12):
            ia = edofs[a]
            K[ia, edofs] += kg[a, :]
        elem_data.append((ni, nj, R, L, E, A))
    K = 0.5 * (K + K.T)
    K_ff = K[free_dofs][:, free_dofs]
    P_f = P[free_dofs]
    try:
        u_f = scipy.linalg.solve(K_ff, P_f, assume_a='sym', check_finite=True, overwrite_a=False, overwrite_b=False)
    except Exception:
        try:
            u_f = _np.linalg.solve(K_ff, P_f)
        except Exception as ex2:
            raise ValueError(f'Failed to solve static equilibrium for reference state: {ex2}')
    if not _np.all(_np.isfinite(u_f)):
        raise ValueError('Non-finite displacements in reference state.')
    u = _np.zeros(n_dof, dtype=float)
    u[free_dofs] = u_f
    for (ni, nj, R, L, E, A) in elem_data:
        edofs = _np.r_[ni * 6 + _np.arange(6), nj * 6 + _np.arange(6)]
        T = _T_from_R(R)
        u_e_g = u[edofs]
        u_e_l = T.T @ u_e_g
        dux = u_e_l[6] - u_e_l[0]
        N = E * A / L * dux
        if abs(N) > tol_solve:
            kg_local = _np.zeros((12, 12), dtype=float)
            idx_y = [1, 5, 7, 11]
            kg_y = _kg_bending_block(N, L)
            for a in range(4):
                for b in range(4):
                    kg_local[idx_y[a], idx_y[b]] += kg_y[a, b]
            idx_z = [2, 4, 8, 10]
            kg_z = _kg_bending_block(N, L)
            for a in range(4):
                for b in range(4):
                    kg_local[idx_z[a], idx_z[b]] += kg_z[a, b]
            kg_global = T @ kg_local @ T.T
            for a in range(12):
                ia = edofs[a]
                Kg[ia, edofs] += kg_global[a, :]
    Kg = 0.5 * (Kg + Kg.T)
    Kg_ff = Kg[free_dofs][:, free_dofs]
    if _np.linalg.norm(Kg_ff, ord='fro') <= tol_solve:
        raise ValueError('Geometric stiffness is near zero under the provided reference load; no buckling load can be computed.')
    try:
        (w, V) = scipy.linalg.eig(K_ff, -Kg_ff, check_finite=True)
    except Exception as ex:
        raise ValueError(f'Generalized eigenvalue solve failed: {ex}')
    if w is None or V is None:
        raise ValueError('Eigenvalue solver returned no result.')
    w = _np.asarray(w)
    V = _np.asarray(V)
    real_part = w.real
    imag_part = w.imag
    real_mask = _np.abs(imag_part) <= tol_imag * _np.maximum(1.0, _np.abs(real_part))
    pos_mask = real_part > tol_pos
    mask = real_mask & pos_mask & _np.isfinite(real_part)
    if not _np.any(mask):
        raise ValueError('No positive real eigenvalue found for the buckling problem.')
    idx_min = _np.argmin(real_part[mask])
    valid_indices = _np.nonzero(mask)[0]
    sel = valid_indices[idx_min]
    lambda_cr = float(real_part[sel])
    phi_f = V[:, sel].real
    phi_full = _np.zeros(n_dof, dtype=float)
    phi_full[free_dofs] = phi_f
    return (lambda_cr, phi_full)