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
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be (N,3)')
    N = node_coords.shape[0]
    Ndof = 6 * N
    K = np.zeros((Ndof, Ndof), dtype=float)
    F = np.zeros(Ndof, dtype=float)
    if nodal_loads:
        for (n, loads) in nodal_loads.items():
            if loads is None:
                continue
            idx = int(n)
            arr = np.asarray(loads, dtype=float)
            if arr.size != 6:
                raise ValueError('Each nodal load must have 6 entries')
            F[6 * idx:6 * idx + 6] = arr
    tol = 1e-12
    for elem in elements:
        ni = int(elem['node_i'])
        nj = int(elem['node_j'])
        xi = node_coords[ni]
        xj = node_coords[nj]
        Lvec = xj - xi
        L = np.linalg.norm(Lvec)
        if L <= 0:
            raise ValueError('Element with zero length')
        ex = Lvec / L
        E = float(elem['E'])
        nu = float(elem['nu'])
        A = float(elem['A'])
        I_y = float(elem['I_y'])
        I_z = float(elem['I_z'])
        J = float(elem['J'])
        G = E / (2.0 * (1.0 + nu))
        a = elem.get('local_z', None)
        if a is None:
            a_vec = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(a_vec, ex)) > 1.0 - 1e-08:
                a_vec = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            a_vec = np.asarray(a, dtype=float)
            if a_vec.size != 3:
                raise ValueError('local_z must be length 3')
            if np.linalg.norm(a_vec) < tol:
                a_vec = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(a_vec, ex)) > 1.0 - 1e-08:
                a_vec = np.array([0.0, 1.0, 0.0], dtype=float)
                if abs(np.dot(a_vec, ex)) > 1.0 - 1e-08:
                    a_vec = np.array([1.0, 0.0, 0.0], dtype=float)
        proj = a_vec - np.dot(a_vec, ex) * ex
        nrm = np.linalg.norm(proj)
        if nrm < tol:
            a_alt = np.array([0.0, 1.0, 0.0], dtype=float)
            proj = a_alt - np.dot(a_alt, ex) * ex
            nrm = np.linalg.norm(proj)
            if nrm < tol:
                a_alt = np.array([1.0, 0.0, 0.0], dtype=float)
                proj = a_alt - np.dot(a_alt, ex) * ex
                nrm = np.linalg.norm(proj)
                if nrm < tol:
                    raise ValueError('Cannot determine local orientation for element')
        ez = proj / nrm
        ey = np.cross(ez, ex)
        ey_n = np.linalg.norm(ey)
        if ey_n < tol:
            raise ValueError('Local basis ill-defined')
        ey = ey / ey_n
        R = np.vstack((ex, ey, ez))
        T = np.kron(np.eye(4), R)
        ke = np.zeros((12, 12), dtype=float)
        ke[0, 0] = ke[6, 6] = E * A / L
        ke[0, 6] = ke[6, 0] = -E * A / L
        ke[3, 3] = ke[9, 9] = G * J / L
        ke[3, 9] = ke[9, 3] = -G * J / L
        k = E * I_z
        L2 = L * L
        L3 = L2 * L
        ind_z = [1, 5, 7, 11]
        k_mat_z = k / L3 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float)
        for (a_i, ii) in enumerate(ind_z):
            for (a_j, jj) in enumerate(ind_z):
                ke[ii, jj] += k_mat_z[a_i, a_j]
        k = E * I_y
        ind_y = [2, 4, 8, 10]
        k_mat_y = k / L3 * np.array([[12.0, -6.0 * L, -12.0, -6.0 * L], [-6.0 * L, 4.0 * L2, 6.0 * L, 2.0 * L2], [-12.0, 6.0 * L, 12.0, 6.0 * L], [-6.0 * L, 2.0 * L2, 6.0 * L, 4.0 * L2]], dtype=float)
        for (a_i, ii) in enumerate(ind_y):
            for (a_j, jj) in enumerate(ind_y):
                ke[ii, jj] += k_mat_y[a_i, a_j]
        kg = T.T @ ke @ T
        edofs = np.concatenate((np.arange(6 * ni, 6 * ni + 6), np.arange(6 * nj, 6 * nj + 6)))
        for a_i in range(12):
            Ii = edofs[a_i]
            for a_j in range(12):
                Jj = edofs[a_j]
                K[Ii, Jj] += kg[a_i, a_j]
    fixed = np.zeros(Ndof, dtype=bool)
    if boundary_conditions:
        for (node, bcvals) in boundary_conditions.items():
            idx = int(node)
            vals = np.asarray(bcvals, dtype=int)
            if vals.size != 6:
                raise ValueError('Each boundary condition entry must have 6 values')
            for k in range(6):
                if int(vals[k]) != 0:
                    fixed[6 * idx + k] = True
    free = ~fixed
    u = np.zeros(Ndof, dtype=float)
    nfree = int(np.count_nonzero(free))
    if nfree > 0:
        Kff = K[np.ix_(free, free)]
        ff = F[free]
        try:
            cond = np.linalg.cond(Kff)
        except Exception:
            cond = np.inf
        if not np.isfinite(cond) or cond >= 1e+16:
            raise ValueError('Free-free stiffness matrix is ill-conditioned or singular')
        u_free = np.linalg.solve(Kff, ff)
        u[free] = u_free
    r = K @ u - F
    return (u, r)