def solve_linear_elastic_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    """
    Small-displacement linear-elastic analysis of a 3D frame using beam elements.
    The function assembles the global stiffness matrix (K) and load vector (P),
    partitions degrees of freedom (DOFs) into free and fixed sets, solves the
    reduced system for displacements at the free DOFs, and computes true support
    reactions at the fixed DOFs.
    Coordinate system: global right-handed Cartesian. Element local axes follow the
    beam axis (local x) with orientation defined via a reference vector.
    Condition number policy: the system is solved only if the free–free stiffness
    submatrix K_ff is well-conditioned (cond(K_ff) < 1e16). Otherwise a ValueError
    is raised.
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
            'local_z' : (3,) array or None
                Optional unit vector to define the local z-direction for transformation
                matrix orientation (must be unit length and not parallel to the beam axis).
    boundary_conditions : dict[int, Sequence[int]]
        Maps node index → 6-element iterable of 0 (free) or 1 (fixed). Omitted nodes ⇒ all DOFs free.
    nodal_loads : dict[int, Sequence[float]]
        Maps node index → 6-element [Fx, Fy, Fz, Mx, My, Mz]. Omitted nodes ⇒ zero loads.
    Returns
    -------
    u : (6*N,) ndarray
        Global displacement vector ordered as [UX, UY, UZ, RX, RY, RZ] for each node
        in sequence. Values are computed at free DOFs; fixed DOFs are zero.
    r : (6*N,) ndarray
        Global reaction force/moment vector with nonzeros only at fixed DOFs.
        Reactions are computed as internal elastic forces minus applied loads at the
        fixed DOFs; free DOFs have zero entries.
    Raises
    ------
    ValueError
        If the free-free submatrix K_ff is ill-conditioned (cond(K_ff) ≥ 1e16).
    Notes
    -----
    """
    N = int(node_coords.shape[0])
    ndof = 6 * N
    K = np.zeros((ndof, ndof), dtype=float)
    P = np.zeros(ndof, dtype=float)
    for (n, load) in (nodal_loads or {}).items():
        if n < 0 or n >= N:
            raise IndexError('nodal_loads contains node index out of range')
        arr = np.asarray(load, dtype=float)
        if arr.shape[0] != 6:
            raise ValueError('Each nodal load must have 6 components [Fx, Fy, Fz, Mx, My, Mz]')
        s = 6 * n
        P[s:s + 6] += arr
    eps = 1e-12
    for el in elements:
        i = int(el['node_i'])
        j = int(el['node_j'])
        if i < 0 or i >= N or j < 0 or (j >= N):
            raise IndexError('Element references node index out of range')
        xi = node_coords[i].astype(float)
        xj = node_coords[j].astype(float)
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= eps:
            raise ValueError('Element length must be positive and finite')
        E = float(el['E'])
        nu = float(el['nu'])
        A = float(el['A'])
        Iy = float(el['I_y'])
        Iz = float(el['I_z'])
        J = float(el['J'])
        G = E / (2.0 * (1.0 + nu))
        ex = dx / L
        local_z = el.get('local_z', None)
        if local_z is not None:
            zref = np.asarray(local_z, dtype=float).reshape(3)
            nz = float(np.linalg.norm(zref))
            if nz <= eps:
                raise ValueError('Provided local_z must be non-zero')
            zref = zref / nz
            ez_tmp = zref - np.dot(zref, ex) * ex
            nzp = float(np.linalg.norm(ez_tmp))
            if nzp <= 1e-08:
                raise ValueError('Provided local_z must not be parallel to the element axis')
            ez = ez_tmp / nzp
        else:
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(ex, ref))) > 1.0 - 1e-06:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
            ez_tmp = ref - np.dot(ref, ex) * ex
            ez = ez_tmp / np.linalg.norm(ez_tmp)
        ey = np.cross(ez, ex)
        ey = ey / np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        R = np.column_stack((ex, ey, ez))
        k_local = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        k_local[0, 0] += k_ax
        k_local[0, 6] += -k_ax
        k_local[6, 0] += -k_ax
        k_local[6, 6] += k_ax
        k_t = G * J / L
        k_local[3, 3] += k_t
        k_local[3, 9] += -k_t
        k_local[9, 3] += -k_t
        k_local[9, 9] += k_t
        a = 12.0 * E * Iz / L ** 3
        b = 6.0 * E * Iz / L ** 2
        c = 4.0 * E * Iz / L
        d = 2.0 * E * Iz / L
        k_local[1, 1] += a
        k_local[1, 5] += b
        k_local[1, 7] += -a
        k_local[1, 11] += b
        k_local[5, 1] += b
        k_local[5, 5] += c
        k_local[5, 7] += -b
        k_local[5, 11] += d
        k_local[7, 1] += -a
        k_local[7, 5] += -b
        k_local[7, 7] += a
        k_local[7, 11] += -b
        k_local[11, 1] += b
        k_local[11, 5] += d
        k_local[11, 7] += -b
        k_local[11, 11] += c
        a = 12.0 * E * Iy / L ** 3
        b = 6.0 * E * Iy / L ** 2
        c = 4.0 * E * Iy / L
        d = 2.0 * E * Iy / L
        k_local[2, 2] += a
        k_local[2, 4] += -b
        k_local[2, 8] += -a
        k_local[2, 10] += -b
        k_local[4, 2] += -b
        k_local[4, 4] += c
        k_local[4, 8] += b
        k_local[4, 10] += d
        k_local[8, 2] += -a
        k_local[8, 4] += b
        k_local[8, 8] += a
        k_local[8, 10] += b
        k_local[10, 2] += -b
        k_local[10, 4] += d
        k_local[10, 8] += b
        k_local[10, 10] += c
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        k_global_e = T @ k_local @ T.T
        dofs = [6 * i + 0, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * j + 0, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5]
        for a_idx in range(12):
            ia = dofs[a_idx]
            Ka_row = K[ia]
            for b_idx in range(12):
                ib = dofs[b_idx]
                Ka_row[ib] += k_global_e[a_idx, b_idx]
    fixed = np.zeros(ndof, dtype=bool)
    if boundary_conditions:
        for (n, bc) in boundary_conditions.items():
            if n < 0 or n >= N:
                raise IndexError('boundary_conditions contains node index out of range')
            arr = np.asarray(bc, dtype=int).reshape(-1)
            if arr.shape[0] != 6:
                raise ValueError('Each boundary condition mask must have 6 entries of 0/1')
            s = 6 * n
            fixed[s:s + 6] = arr.astype(bool)
    free = ~fixed
    u = np.zeros(ndof, dtype=float)
    n_free = int(np.count_nonzero(free))
    if n_free > 0:
        K_ff = K[np.ix_(free, free)]
        P_f = P[free]
        cond = np.linalg.cond(K_ff) if K_ff.size > 0 else 0.0
        if not np.isfinite(cond) or cond >= 1e+16:
            raise ValueError('Ill-conditioned free–free stiffness matrix (cond ≥ 1e16)')
        u_f = np.linalg.solve(K_ff, P_f) if K_ff.size > 0 else np.zeros(0, dtype=float)
        u[free] = u_f
        r_full = K @ u - P
        r = np.zeros(ndof, dtype=float)
        r[fixed] = r_full[fixed]
    else:
        r = np.zeros(ndof, dtype=float)
        r[fixed] = -P[fixed]
    return (u, r)