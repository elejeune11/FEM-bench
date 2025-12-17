def MSA_3D_linear_elastic_CC0_H6_T3(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    """
    Small-displacement linear-elastic analysis of a 3D frame using beam elements.
    Assumes global Cartesian coordinates and right-hand rule for orientation.
    the degrees of freedom (DOFs) into free and fixed sets, and solves the system
    for nodal displacements and support reactions.
    The system is solved using a partitioned approach. Displacements are computed
    at free DOFs, and true reaction forces (including contributions from both
    stiffness and applied loads) are computed at fixed DOFs. The system is only
    solved if the free-free stiffness matrix is well-conditioned
    (i.e., condition number < 1e16). If the matrix is ill-conditioned or singular,
    a ValueError is raised.
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
    u : (6 * N,) ndarray
        Global displacement vector. Entries are ordered as [UX, UY, UZ, RX, RY, RZ] for each node.
        Displacements are computed only at free DOFs; fixed DOFs are set to zero.
    r : (6 * N,) ndarray
        Global reaction force and moment vector. Nonzero values are present only at fixed DOFs
        and reflect the net support reactions, computed as internal elastic forces minus applied loads.
    Raises
    ------
    ValueError
        If the free-free stiffness matrix is ill-conditioned and the system cannot be reliably solved.
    """
    coords = np.asarray(node_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('node_coords must be an (N, 3) array.')
    n_nodes = coords.shape[0]
    ndof = 6 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    f = np.zeros(ndof, dtype=float)
    if nodal_loads:
        for n, loads in nodal_loads.items():
            if n < 0 or n >= n_nodes:
                raise IndexError('nodal_loads contains an invalid node index.')
            arr = np.asarray(loads, dtype=float).reshape(-1)
            if arr.size != 6:
                raise ValueError('Each nodal load entry must have 6 components [Fx, Fy, Fz, Mx, My, Mz].')
            base = 6 * n
            f[base:base + 6] += arr

    def dof_idx(node_index: int):
        base = 6 * node_index
        return np.arange(base, base + 6, dtype=int)
    for e in elements:
        ni = int(e['node_i'])
        nj = int(e['node_j'])
        if ni < 0 or ni >= n_nodes or nj < 0 or (nj >= n_nodes):
            raise IndexError('Element references an invalid node index.')
        xi = coords[ni]
        xj = coords[nj]
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element has zero or invalid length.')
        ex = dx / L
        lz_in = e.get('local_z', None)
        if lz_in is None:
            global_z = np.array([0.0, 0.0, 1.0], dtype=float)
            global_y = np.array([0.0, 1.0, 0.0], dtype=float)
            if abs(np.dot(ex, global_z)) > 1.0 - 1e-10:
                lz = global_y
            else:
                lz = global_z
        else:
            lz = np.asarray(lz_in, dtype=float).reshape(-1)
            if lz.size != 3:
                raise ValueError('local_z must be a 3-element vector.')
            nrm_lz = float(np.linalg.norm(lz))
            if nrm_lz == 0.0 or not np.isfinite(nrm_lz):
                raise ValueError('local_z must be a nonzero finite vector.')
            lz = lz / nrm_lz
            if abs(np.dot(lz, ex)) >= 1.0 - 1e-10:
                raise ValueError('local_z must not be parallel to the element axis.')
        ey = np.cross(lz, ex)
        n_ey = float(np.linalg.norm(ey))
        if n_ey < 1e-14:
            raise ValueError('Failed to construct a valid local coordinate system.')
        ey = ey / n_ey
        ez = np.cross(ex, ey)
        R = np.column_stack((ex, ey, ez))
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['I_y'])
        Iz = float(e['I_z'])
        J = float(e['J'])
        G = E / (2.0 * (1.0 + nu))
        k_loc = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        k_loc[0, 0] = k_ax
        k_loc[0, 6] = -k_ax
        k_loc[6, 0] = -k_ax
        k_loc[6, 6] = k_ax
        k_to = G * J / L
        k_loc[3, 3] = k_to
        k_loc[3, 9] = -k_to
        k_loc[9, 3] = -k_to
        k_loc[9, 9] = k_to
        EIz = E * Iz
        L2 = L * L
        L3 = L2 * L
        kbz = EIz / L3 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float)
        idx_bz = [1, 5, 7, 11]
        for a in range(4):
            ia = idx_bz[a]
            for b in range(4):
                ib = idx_bz[b]
                k_loc[ia, ib] += kbz[a, b]
        EIy = E * Iy
        kby = EIy / L3 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float)
        idx_by = [2, 4, 8, 10]
        for a in range(4):
            ia = idx_by[a]
            for b in range(4):
                ib = idx_by[b]
                k_loc[ia, ib] += kby[a, b]
        k_gl = T @ k_loc @ T.T
        idx_i = dof_idx(ni)
        idx_j = dof_idx(nj)
        elem_dofs = np.concatenate((idx_i, idx_j))
        K[np.ix_(elem_dofs, elem_dofs)] += k_gl
    fixed = np.zeros(ndof, dtype=bool)
    if boundary_conditions:
        for n, bc in boundary_conditions.items():
            if n < 0 or n >= n_nodes:
                raise IndexError('boundary_conditions contains an invalid node index.')
            arr = np.asarray(bc, dtype=int).reshape(-1)
            if arr.size != 6:
                raise ValueError('Each boundary condition entry must have 6 components (0 free, 1 fixed).')
            base = 6 * n
            fixed[base:base + 6] = arr.astype(bool)
    free = ~fixed
    u = np.zeros(ndof, dtype=float)
    r = np.zeros(ndof, dtype=float)
    n_free = int(np.count_nonzero(free))
    if n_free > 0:
        K_ff = K[np.ix_(free, free)]
        f_f = f[free]
        try:
            cond = np.linalg.cond(K_ff)
        except np.linalg.LinAlgError as exc:
            raise ValueError('Free-free stiffness matrix is singular or ill-conditioned.') from exc
        if not np.isfinite(cond) or cond >= 1e+16:
            raise ValueError('Free-free stiffness matrix is ill-conditioned; analysis cannot proceed reliably.')
        uf = np.linalg.solve(K_ff, f_f)
        u[free] = uf
        if np.any(fixed):
            K_cf = K[np.ix_(fixed, free)]
            f_c = f[fixed]
            r_c = K_cf @ uf - f_c
            r[fixed] = r_c
    elif np.any(fixed):
        r[fixed] = -f[fixed]
    return (u, r)