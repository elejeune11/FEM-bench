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
    N = int(node_coords.shape[0])
    ndof = 6 * N
    K = np.zeros((ndof, ndof), dtype=float)
    f = np.zeros(ndof, dtype=float)
    if nodal_loads is not None:
        for n, loads in nodal_loads.items():
            n = int(n)
            lv = np.asarray(loads, dtype=float).reshape(-1)
            lv6 = np.zeros(6, dtype=float)
            m = min(6, lv.size)
            lv6[:m] = lv[:m]
            i0 = 6 * n
            f[i0:i0 + 6] += lv6
    for e in elements:
        i = int(e['node_i'])
        j = int(e['node_j'])
        xi = np.asarray(node_coords[i], dtype=float)
        xj = np.asarray(node_coords[j], dtype=float)
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element has zero or invalid length.')
        ex = dx / L
        z_ref = e.get('local_z', None)
        if z_ref is None:
            z_candidate = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(ex, z_candidate))) > 1.0 - 1e-10:
                z_candidate = np.array([0.0, 1.0, 0.0], dtype=float)
            z_ref = z_candidate
        else:
            z_ref = np.asarray(z_ref, dtype=float)
            nz = float(np.linalg.norm(z_ref))
            if nz == 0.0:
                z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                z_ref = z_ref / nz
            if abs(float(np.dot(ex, z_ref))) > 1.0 - 1e-10:
                z_candidate = np.array([0.0, 0.0, 1.0], dtype=float)
                if abs(float(np.dot(ex, z_candidate))) > 1.0 - 1e-10:
                    z_candidate = np.array([0.0, 1.0, 0.0], dtype=float)
                z_ref = z_candidate
        ez_tmp = z_ref - float(np.dot(z_ref, ex)) * ex
        nez = float(np.linalg.norm(ez_tmp))
        if nez < 1e-14:
            z_candidate = np.array([0.0, 1.0, 0.0], dtype=float)
            ez_tmp = z_candidate - float(np.dot(z_candidate, ex)) * ex
            nez = float(np.linalg.norm(ez_tmp))
            if nez < 1e-14:
                z_candidate = np.array([1.0, 0.0, 0.0], dtype=float)
                ez_tmp = z_candidate - float(np.dot(z_candidate, ex)) * ex
                nez = float(np.linalg.norm(ez_tmp))
        ez_prime = ez_tmp / nez
        ey = np.cross(ez_prime, ex)
        ney = float(np.linalg.norm(ey))
        if ney < 1e-14:
            tmp = np.array([0.0, 1.0, 0.0], dtype=float)
            if abs(float(np.dot(ex, tmp))) > 1.0 - 1e-10:
                tmp = np.array([1.0, 0.0, 0.0], dtype=float)
            ez_prime = tmp - float(np.dot(tmp, ex)) * ex
            ez_prime = ez_prime / float(np.linalg.norm(ez_prime))
            ey = np.cross(ez_prime, ex)
            ney = float(np.linalg.norm(ey))
        ey = ey / ney
        ez = np.cross(ex, ey)
        R = np.vstack((ex, ey, ez))
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['I_y'])
        Iz = float(e['I_z'])
        J = float(e['J'])
        G = E / (2.0 * (1.0 + nu))
        k_local = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        k_local[0, 0] += k_ax
        k_local[0, 6] -= k_ax
        k_local[6, 0] -= k_ax
        k_local[6, 6] += k_ax
        k_t = G * J / L
        k_local[3, 3] += k_t
        k_local[3, 9] -= k_t
        k_local[9, 3] -= k_t
        k_local[9, 9] += k_t
        Kbz = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L]], dtype=float) * (E * Iz / L ** 3)
        idx_bz = [1, 5, 7, 11]
        for a in range(4):
            ia = idx_bz[a]
            for b in range(4):
                ib = idx_bz[b]
                k_local[ia, ib] += Kbz[a, b]
        Kby = np.array([[12.0, -6.0 * L, -12.0, -6.0 * L], [-6.0 * L, 4.0 * L * L, 6.0 * L, 2.0 * L * L], [-12.0, 6.0 * L, 12.0, 6.0 * L], [-6.0 * L, 2.0 * L * L, 6.0 * L, 4.0 * L * L]], dtype=float) * (E * Iy / L ** 3)
        idx_by = [2, 4, 8, 10]
        for a in range(4):
            ia = idx_by[a]
            for b in range(4):
                ib = idx_by[b]
                k_local[ia, ib] += Kby[a, b]
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        k_global = T.T @ k_local @ T
        gidx = np.array([6 * i + 0, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * j + 0, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5], dtype=int)
        K[np.ix_(gidx, gidx)] += k_global
    K = 0.5 * (K + K.T)
    free_mask = np.ones(ndof, dtype=bool)
    if boundary_conditions is not None:
        for n, bc in boundary_conditions.items():
            n = int(n)
            bcarr = np.asarray(bc, dtype=int).reshape(-1)
            pad = np.zeros(6, dtype=int)
            m = min(6, bcarr.size)
            pad[:m] = bcarr[:m]
            for k in range(6):
                if pad[k] != 0:
                    free_mask[6 * n + k] = False
    free_dof = np.nonzero(free_mask)[0]
    fixed_dof = np.nonzero(~free_mask)[0]
    u = np.zeros(ndof, dtype=float)
    if free_dof.size > 0:
        Kff = K[np.ix_(free_dof, free_dof)]
        ff = f[free_dof]
        cond = np.linalg.cond(Kff)
        if not np.isfinite(cond) or cond >= 1e+16:
            raise ValueError('Ill-conditioned or singular free-free stiffness matrix.')
        uf = np.linalg.solve(Kff, ff)
        u[free_dof] = uf
    r = K @ u - f
    if free_dof.size > 0:
        r[free_dof] = 0.0
    return (u, r)