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

    def _compute_rotation(xi: np.ndarray, xj: np.ndarray, local_z_hint: np.ndarray | None) -> np.ndarray:
        ex = xj - xi
        L = np.linalg.norm(ex)
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element has zero or invalid length.')
        ex = ex / L
        z_global = np.array([0.0, 0.0, 1.0])
        y_global = np.array([0.0, 1.0, 0.0])
        if local_z_hint is None:
            if abs(np.dot(ex, z_global)) < 1.0 - 1e-08:
                vref = z_global
            else:
                vref = y_global
        else:
            vref = np.asarray(local_z_hint, dtype=float)
            nref = np.linalg.norm(vref)
            if not np.isfinite(nref) or nref <= 0.0:
                if abs(np.dot(ex, z_global)) < 1.0 - 1e-08:
                    vref = z_global
                else:
                    vref = y_global
            else:
                vref = vref / nref
        ey = np.cross(vref, ex)
        ny = np.linalg.norm(ey)
        if ny <= 1e-12 or not np.isfinite(ny):
            alt = y_global if abs(np.dot(ex, z_global)) >= 1.0 - 1e-08 else z_global
            ey = np.cross(alt, ex)
            ny = np.linalg.norm(ey)
            if ny <= 1e-12 or not np.isfinite(ny):
                t = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(ex, t)) > 0.9:
                    t = np.array([0.0, 1.0, 0.0])
                ey = np.cross(t, ex)
                ny = np.linalg.norm(ey)
                if ny <= 1e-12 or not np.isfinite(ny):
                    raise ValueError('Failed to construct local element axes.')
        ey = ey / ny
        ez = np.cross(ex, ey)
        R = np.vstack((ex, ey, ez)).T
        return (R, L)

    def _local_stiffness(E: float, G: float, A: float, Iy: float, Iz: float, J: float, L: float) -> np.ndarray:
        k = np.zeros((12, 12), dtype=float)
        EA_L = E * A / L
        k[0, 0] = EA_L
        k[0, 6] = -EA_L
        k[6, 0] = -EA_L
        k[6, 6] = EA_L
        GJ_L = G * J / L
        k[3, 3] = GJ_L
        k[3, 9] = -GJ_L
        k[9, 3] = -GJ_L
        k[9, 9] = GJ_L
        EIz = E * Iz
        c1 = 12.0 * EIz / L ** 3
        c2 = 6.0 * EIz / L ** 2
        c3 = 4.0 * EIz / L
        c4 = 2.0 * EIz / L
        k[1, 1] = c1
        k[1, 5] = c2
        k[1, 7] = -c1
        k[1, 11] = c2
        k[5, 1] = c2
        k[5, 5] = c3
        k[5, 7] = -c2
        k[5, 11] = c4
        k[7, 1] = -c1
        k[7, 5] = -c2
        k[7, 7] = c1
        k[7, 11] = -c2
        k[11, 1] = c2
        k[11, 5] = c4
        k[11, 7] = -c2
        k[11, 11] = c3
        EIy = E * Iy
        d1 = 12.0 * EIy / L ** 3
        d2 = 6.0 * EIy / L ** 2
        d3 = 4.0 * EIy / L
        d4 = 2.0 * EIy / L
        k[2, 2] = d1
        k[2, 4] = -d2
        k[2, 8] = -d1
        k[2, 10] = -d2
        k[4, 2] = -d2
        k[4, 4] = d3
        k[4, 8] = d2
        k[4, 10] = d4
        k[8, 2] = -d1
        k[8, 4] = d2
        k[8, 8] = d1
        k[8, 10] = d2
        k[10, 2] = -d2
        k[10, 4] = d4
        k[10, 8] = d2
        k[10, 10] = d3
        return k
    N = int(node_coords.shape[0])
    ndof = 6 * N
    K = np.zeros((ndof, ndof), dtype=float)
    f = np.zeros(ndof, dtype=float)
    for e in elements:
        ni = int(e['node_i'])
        nj = int(e['node_j'])
        xi = np.asarray(node_coords[ni], dtype=float)
        xj = np.asarray(node_coords[nj], dtype=float)
        local_z = e.get('local_z', None)
        (R, L) = _compute_rotation(xi, xj, None if local_z is None else np.asarray(local_z, dtype=float))
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['I_y'])
        Iz = float(e['I_z'])
        J = float(e['J'])
        G = E / (2.0 * (1.0 + nu))
        k_loc = _local_stiffness(E, G, A, Iy, Iz, J, L)
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        k_gl = T.T @ k_loc @ T
        dofs = np.array([6 * ni + 0, 6 * ni + 1, 6 * ni + 2, 6 * ni + 3, 6 * ni + 4, 6 * ni + 5, 6 * nj + 0, 6 * nj + 1, 6 * nj + 2, 6 * nj + 3, 6 * nj + 4, 6 * nj + 5], dtype=int)
        K[np.ix_(dofs, dofs)] += k_gl
    for (node, loads) in (nodal_loads or {}).items():
        idx0 = 6 * int(node)
        arr = np.asarray(loads, dtype=float).reshape(6)
        f[idx0:idx0 + 6] += arr
    is_fixed = np.zeros(ndof, dtype=bool)
    if boundary_conditions is not None:
        for (node, bc) in boundary_conditions.items():
            idx0 = 6 * int(node)
            flags = np.asarray(bc, dtype=int).reshape(6)
            flags = flags.astype(bool)
            is_fixed[idx0:idx0 + 6] = flags
    free = np.where(~is_fixed)[0]
    fixed = np.where(is_fixed)[0]
    u = np.zeros(ndof, dtype=float)
    if free.size > 0:
        K_ff = K[np.ix_(free, free)]
        f_f = f[free]
        try:
            cond = np.linalg.cond(K_ff)
        except np.linalg.LinAlgError as err:
            raise ValueError('Failed to assess conditioning of the free-free stiffness matrix.') from err
        if not np.isfinite(cond) or cond >= 1e+16:
            raise ValueError('Ill-conditioned or singular free-free stiffness matrix.')
        try:
            u_f = np.linalg.solve(K_ff, f_f)
        except np.linalg.LinAlgError as err:
            raise ValueError('Ill-conditioned or singular free-free stiffness matrix.') from err
        u[free] = u_f
    r = K @ u - f
    r[~is_fixed] = 0.0
    return (u, r)