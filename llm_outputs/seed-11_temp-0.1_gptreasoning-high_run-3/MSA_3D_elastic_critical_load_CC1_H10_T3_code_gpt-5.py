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
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an (N,3) array of floats.')
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    required_keys = {'node_i', 'node_j', 'E', 'nu', 'A', 'I_y', 'I_z', 'J'}
    if not hasattr(elements, '__iter__'):
        raise ValueError('elements must be an iterable of dicts.')
    elements_list = []
    for e in elements:
        if not isinstance(e, dict):
            raise ValueError('Each element must be a dict.')
        if not required_keys.issubset(set(e.keys())):
            raise ValueError(f'Each element dict must contain keys: {sorted(required_keys)}')
        i = int(e['node_i'])
        j = int(e['node_j'])
        if not (0 <= i < n_nodes and 0 <= j < n_nodes and (i != j)):
            raise ValueError('Element node indices must be valid and distinct within node_coords.')
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['I_y'])
        Iz = float(e['I_z'])
        J = float(e['J'])
        if E <= 0 or A <= 0 or Iy <= 0 or (Iz <= 0) or (J <= 0):
            raise ValueError('Element properties E, A, I_y, I_z, J must be positive.')
        local_z = e.get('local_z', None)
        if local_z is not None:
            local_z = np.asarray(local_z, dtype=float).reshape(3)
        elements_list.append({'node_i': i, 'node_j': j, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': local_z})
    fixed = np.zeros(ndof, dtype=bool)
    if boundary_conditions is not None:
        if not isinstance(boundary_conditions, dict):
            raise ValueError('boundary_conditions must be a dict mapping node index to 6 fixed/free flags.')
        for n, flags in boundary_conditions.items():
            if not 0 <= int(n) < n_nodes:
                raise ValueError('boundary_conditions: invalid node index.')
            f = np.asarray(flags, dtype=int).flatten()
            if f.size != 6 or not np.all((f == 0) | (f == 1)):
                raise ValueError('boundary_conditions: each node must map to 6 values of 0 or 1.')
            for k in range(6):
                fixed[6 * int(n) + k] = bool(f[k])
    P = np.zeros(ndof, dtype=float)
    if nodal_loads is not None:
        if not isinstance(nodal_loads, dict):
            raise ValueError('nodal_loads must be a dict mapping node index to 6 loads [Fx,Fy,Fz,Mx,My,Mz].')
        for n, loads in nodal_loads.items():
            if not 0 <= int(n) < n_nodes:
                raise ValueError('nodal_loads: invalid node index.')
            Ls = np.asarray(loads, dtype=float).flatten()
            if Ls.size != 6:
                raise ValueError('nodal_loads: each node must map to 6 load components.')
            P[6 * int(n):6 * int(n) + 6] = Ls

    def build_local_axes(xi: np.ndarray, xj: np.ndarray, local_z_hint: np.ndarray | None) -> np.ndarray:
        vi = np.asarray(xi, dtype=float)
        vj = np.asarray(xj, dtype=float)
        v = vj - vi
        L = np.linalg.norm(v)
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element has zero or invalid length.')
        x_hat = v / L
        if local_z_hint is None:
            z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if np.linalg.norm(np.cross(x_hat, z_ref)) < 1e-12:
                z_ref = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            z_ref = np.array(local_z_hint, dtype=float)
        z_perp = z_ref - np.dot(z_ref, x_hat) * x_hat
        nrm = np.linalg.norm(z_perp)
        if nrm < 1e-12:
            alt = np.array([0.0, 0.0, 1.0], dtype=float)
            if np.linalg.norm(np.cross(x_hat, alt)) < 1e-12:
                alt = np.array([0.0, 1.0, 0.0], dtype=float)
            z_perp = alt - np.dot(alt, x_hat) * x_hat
            nrm = np.linalg.norm(z_perp)
            if nrm < 1e-12:
                raise ValueError('Could not determine a valid local z-axis not parallel to element axis.')
        z_hat = z_perp / nrm
        y_hat = np.cross(z_hat, x_hat)
        y_hat /= np.linalg.norm(y_hat)
        z_hat = np.cross(x_hat, y_hat)
        z_hat /= np.linalg.norm(z_hat)
        Q = np.column_stack((x_hat, y_hat, z_hat))
        return (Q, L)

    def elastic_stiffness_local(E: float, G: float, A: float, Iy: float, Iz: float, J: float, L: float) -> np.ndarray:
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
        EIz = E * Iz
        L2 = L * L
        L3 = L2 * L
        kz = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float) * (EIz / L3)
        idx_y = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                k[idx_y[a], idx_y[b]] += kz[a, b]
        EIy = E * Iy
        ky = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float) * (EIy / L3)
        idx_z = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                k[idx_z[a], idx_z[b]] += ky[a, b]
        return k

    def geometric_stiffness_local(N: float, L: float) -> np.ndarray:
        kg = np.zeros((12, 12), dtype=float)
        if abs(L) <= 0.0:
            return kg
        factor = N / (30.0 * L)
        base = np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L * L, -3.0 * L, -1.0 * L * L], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -1.0 * L * L, -3.0 * L, 4.0 * L * L]], dtype=float) * factor
        idx_y = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                kg[idx_y[a], idx_y[b]] += base[a, b]
        idx_z = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                kg[idx_z[a], idx_z[b]] += base[a, b]
        return kg
    K = np.zeros((ndof, ndof), dtype=float)
    elem_data = []
    for e in elements_list:
        i = e['node_i']
        j = e['node_j']
        xi = node_coords[i]
        xj = node_coords[j]
        Q, L = build_local_axes(xi, xj, e['local_z'])
        G = e['E'] / (2.0 * (1.0 + e['nu']))
        k_loc = elastic_stiffness_local(e['E'], G, e['A'], e['I_y'], e['I_z'], e['J'], L)
        T = np.zeros((12, 12), dtype=float)
        QT = Q.T
        T[0:3, 0:3] = QT
        T[3:6, 3:6] = QT
        T[6:9, 6:9] = QT
        T[9:12, 9:12] = QT
        k_glob = T.T @ k_loc @ T
        dof_indices = np.array([6 * i + 0, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * j + 0, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5], dtype=int)
        for a in range(12):
            ia = dof_indices[a]
            Ka = K[ia]
            for b in range(12):
                ib = dof_indices[b]
                Ka[ib] += k_glob[a, b]
        elem_data.append((i, j, Q, L, T, e))
    free = np.where(~fixed)[0]
    if free.size == 0:
        raise ValueError('All DOFs are constrained; no free DOFs to analyze.')
    K_ff = K[np.ix_(free, free)]
    P_f = P[free]
    try:
        condK = np.linalg.cond(K_ff)
    except Exception:
        condK = np.inf
    if not np.isfinite(condK) or condK > 1000000000000.0:
        raise ValueError('Reduced stiffness matrix is singular or ill-conditioned. Check boundary conditions.')
    try:
        u_f = scipy.linalg.solve(K_ff, P_f, assume_a='sym')
    except Exception:
        try:
            u_f = scipy.linalg.lu_solve(scipy.linalg.lu_factor(K_ff), P_f)
        except Exception as ex:
            raise ValueError('Failed to solve the static equilibrium equations. Matrix may be singular.') from ex
    u = np.zeros(ndof, dtype=float)
    u[free] = u_f
    Kg = np.zeros((ndof, ndof), dtype=float)
    for i, j, Q, L, T, e in elem_data:
        dof_idx = np.array([6 * i + 0, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * j + 0, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5], dtype=int)
        u_e_global = u[dof_idx]
        u_e_local = T @ u_e_global
        N = e['E'] * e['A'] / L * (u_e_local[6] - u_e_local[0])
        kg_loc = geometric_stiffness_local(N, L)
        kg_glob = T.T @ kg_loc @ T
        for a in range(12):
            ia = dof_idx[a]
            Kga = Kg[ia]
            for b in range(12):
                ib = dof_idx[b]
                Kga[ib] += kg_glob[a, b]
    Kg_ff = Kg[np.ix_(free, free)]
    if not np.any(np.abs(Kg_ff) > 0):
        raise ValueError('Geometric stiffness matrix is zero; cannot perform buckling analysis under the provided reference load.')
    try:
        evals, evecs = scipy.linalg.eig(K_ff, -Kg_ff)
    except Exception as ex:
        raise ValueError('Failed to solve the generalized eigenvalue problem.') from ex
    evals = np.asarray(evals)
    evecs = np.asarray(evecs)
    imag_abs = np.abs(np.imag(evals))
    real_abs = np.abs(np.real(evals))
    tol_imag = 1e-06
    is_real = (imag_abs <= tol_imag * np.maximum(1.0, real_abs)) | (imag_abs <= 1e-12)
    real_evals = np.real(evals[is_real])
    real_evecs = np.real(evecs[:, is_real]) if real_evals.size > 0 else np.empty((K_ff.shape[0], 0))
    pos_mask = real_evals > 0.0
    if not np.any(pos_mask):
        raise ValueError('No positive real eigenvalue found; the structure may not buckle under the given reference load.')
    pos_evals = real_evals[pos_mask]
    pos_evecs = real_evecs[:, pos_mask]
    idx_min = np.argmin(pos_evals)
    lam = float(pos_evals[idx_min])
    mode_f = pos_evecs[:, idx_min]
    mode_global = np.zeros(ndof, dtype=float)
    mode_global[free] = mode_f
    return (lam, mode_global)