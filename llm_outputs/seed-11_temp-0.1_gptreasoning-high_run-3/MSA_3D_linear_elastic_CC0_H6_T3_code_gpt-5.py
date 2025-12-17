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
    import numpy as np
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an (N, 3) array.')
    N = node_coords.shape[0]
    ndof = 6 * N
    K = np.zeros((ndof, ndof), dtype=float)
    f = np.zeros(ndof, dtype=float)
    for e in elements:
        i = int(e['node_i'])
        j = int(e['node_j'])
        if not (0 <= i < N and 0 <= j < N):
            raise ValueError('Element node indices out of bounds.')
        xi = node_coords[i]
        xj = node_coords[j]
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element has zero or invalid length.')
        ex = dx / L
        z_ref = e.get('local_z', None)
        if z_ref is None:
            global_z = np.array([0.0, 0.0, 1.0], dtype=float)
            global_y = np.array([0.0, 1.0, 0.0], dtype=float)
            if abs(float(np.dot(ex, global_z))) > 1.0 - 1e-12:
                z_ref = global_y
            else:
                z_ref = global_z
        else:
            z_ref = np.asarray(z_ref, dtype=float)
            if z_ref.shape != (3,):
                raise ValueError('local_z must be a 3-element vector.')
            nz = np.linalg.norm(z_ref)
            if not np.isfinite(nz) or nz == 0.0:
                raise ValueError('local_z must be a nonzero vector.')
            z_ref = z_ref / nz
            if abs(float(np.dot(z_ref, ex))) > 1.0 - 1e-12:
                raise ValueError('Provided local_z is parallel to the element axis.')
        y_temp = np.cross(z_ref, ex)
        ny = np.linalg.norm(y_temp)
        if ny <= 1e-15:
            raise ValueError('local_z leads to a degenerate local coordinate system.')
        ey = y_temp / ny
        ez = np.cross(ex, ey)
        R = np.vstack((ex, ey, ez))
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['I_y'])
        Iz = float(e['I_z'])
        J = float(e['J'])
        G = E / (2.0 * (1.0 + nu))
        k_loc = np.zeros((12, 12), dtype=float)
        EA_L = E * A / L
        k_loc[0, 0] = EA_L
        k_loc[0, 6] = -EA_L
        k_loc[6, 0] = -EA_L
        k_loc[6, 6] = EA_L
        GJ_L = G * J / L
        k_loc[3, 3] = GJ_L
        k_loc[3, 9] = -GJ_L
        k_loc[9, 3] = -GJ_L
        k_loc[9, 9] = GJ_L
        EIz = E * Iz
        a = 12.0 * EIz / L ** 3
        b = 6.0 * EIz / L ** 2
        c = 4.0 * EIz / L
        d = 2.0 * EIz / L
        iy, irz, jy, jrz = (1, 5, 7, 11)
        k_loc[iy, iy] += a
        k_loc[iy, irz] += b
        k_loc[iy, jy] += -a
        k_loc[iy, jrz] += b
        k_loc[irz, iy] += b
        k_loc[irz, irz] += c
        k_loc[irz, jy] += -b
        k_loc[irz, jrz] += d
        k_loc[jy, iy] += -a
        k_loc[jy, irz] += -b
        k_loc[jy, jy] += a
        k_loc[jy, jrz] += -b
        k_loc[jrz, iy] += b
        k_loc[jrz, irz] += d
        k_loc[jrz, jy] += -b
        k_loc[jrz, jrz] += c
        EIy = E * Iy
        a = 12.0 * EIy / L ** 3
        b = 6.0 * EIy / L ** 2
        c = 4.0 * EIy / L
        d = 2.0 * EIy / L
        iz, iry, jz, jry = (2, 4, 8, 10)
        k_loc[iz, iz] += a
        k_loc[iz, iry] += b
        k_loc[iz, jz] += -a
        k_loc[iz, jry] += b
        k_loc[iry, iz] += b
        k_loc[iry, iry] += c
        k_loc[iry, jz] += -b
        k_loc[iry, jry] += d
        k_loc[jz, iz] += -a
        k_loc[jz, iry] += -b
        k_loc[jz, jz] += a
        k_loc[jz, jry] += -b
        k_loc[jry, iz] += b
        k_loc[jry, iry] += d
        k_loc[jry, jz] += -b
        k_loc[jry, jry] += c
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        k_glob = T.T @ k_loc @ T
        dofs = list(range(6 * i, 6 * i + 6)) + list(range(6 * j, 6 * j + 6))
        dofs = np.array(dofs, dtype=int)
        K[np.ix_(dofs, dofs)] += k_glob
    for n, load in (nodal_loads or {}).items():
        n = int(n)
        if not 0 <= n < N:
            raise ValueError('Load applied to invalid node index.')
        vec = np.asarray(load, dtype=float).reshape(-1)
        if vec.size != 6:
            raise ValueError('Each nodal load must have 6 components [Fx, Fy, Fz, Mx, My, Mz].')
        f[6 * n:6 * n + 6] += vec
    is_fixed = np.zeros(ndof, dtype=bool)
    if boundary_conditions is not None:
        for n, bc in boundary_conditions.items():
            n = int(n)
            if not 0 <= n < N:
                raise ValueError('Boundary condition specified for invalid node index.')
            bc_arr = np.asarray(bc, dtype=int).reshape(-1)
            if bc_arr.size != 6:
                raise ValueError('Each boundary condition must have 6 entries (0 free, 1 fixed).')
            is_fixed[6 * n:6 * n + 6] = bc_arr.astype(bool)
    free_dofs = np.where(~is_fixed)[0]
    fixed_dofs = np.where(is_fixed)[0]
    u = np.zeros(ndof, dtype=float)
    if free_dofs.size > 0:
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        f_f = f[free_dofs]
        try:
            cond = np.linalg.cond(K_ff)
        except np.linalg.LinAlgError:
            cond = np.inf
        if not np.isfinite(cond) or cond >= 1e+16:
            raise ValueError('The free-free stiffness matrix is ill-conditioned or singular; analysis aborted.')
        u_f = np.linalg.solve(K_ff, f_f)
        u[free_dofs] = u_f
    r = K @ u - f
    if free_dofs.size > 0:
        r[free_dofs] = 0.0
    return (u, r)