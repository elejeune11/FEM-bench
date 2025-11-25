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
    if nodal_loads is not None:
        for (n, loads) in nodal_loads.items():
            idx0 = 6 * int(n)
            vec = np.asarray(loads, dtype=float).reshape(-1)
            if vec.size != 6:
                raise ValueError('Each nodal load must have 6 components.')
            P[idx0:idx0 + 6] += vec
    for e in elements:
        ni = int(e['node_i'])
        nj = int(e['node_j'])
        xi = np.asarray(node_coords[ni], dtype=float)
        xj = np.asarray(node_coords[nj], dtype=float)
        d = xj - xi
        L = float(np.linalg.norm(d))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element length must be positive and finite.')
        ex = d / L
        ref = e.get('local_z', None)
        if ref is None:
            if abs(ex[2]) < 0.9:
                ref_vec = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                ref_vec = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            ref_vec = np.asarray(ref, dtype=float).reshape(3)
            nrm = np.linalg.norm(ref_vec)
            if nrm == 0.0 or not np.isfinite(nrm):
                if abs(ex[2]) < 0.9:
                    ref_vec = np.array([0.0, 0.0, 1.0], dtype=float)
                else:
                    ref_vec = np.array([0.0, 1.0, 0.0], dtype=float)
            else:
                ref_vec = ref_vec / nrm
        temp = ref_vec - np.dot(ref_vec, ex) * ex
        ntemp = np.linalg.norm(temp)
        if ntemp < 1e-12:
            if abs(ex[2]) < 0.9:
                ref_vec = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                ref_vec = np.array([0.0, 1.0, 0.0], dtype=float)
            temp = ref_vec - np.dot(ref_vec, ex) * ex
            ntemp = np.linalg.norm(temp)
            if ntemp < 1e-12:
                alt = np.array([1.0, 0.0, 0.0], dtype=float)
                if abs(np.dot(alt, ex)) > 0.9:
                    alt = np.array([0.0, 1.0, 0.0], dtype=float)
                temp = alt - np.dot(alt, ex) * ex
                ntemp = np.linalg.norm(temp)
                if ntemp < 1e-12:
                    raise ValueError('Failed to determine element local axes; check geometry.')
        ez = temp / ntemp
        ey = np.cross(ez, ex)
        ney = np.linalg.norm(ey)
        if ney == 0.0:
            raise ValueError('Failed to build orthonormal triad for element.')
        ey = ey / ney
        Lambda = np.stack([ex, ey, ez], axis=0)
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['I_y'])
        Iz = float(e['I_z'])
        J = float(e['J'])
        G = E / (2.0 * (1.0 + nu))
        Kloc = np.zeros((12, 12), dtype=float)
        EA_L = E * A / L
        Kloc[0, 0] = EA_L
        Kloc[0, 6] = -EA_L
        Kloc[6, 0] = -EA_L
        Kloc[6, 6] = EA_L
        GJ_L = G * J / L
        Kloc[3, 3] = GJ_L
        Kloc[3, 9] = -GJ_L
        Kloc[9, 3] = -GJ_L
        Kloc[9, 9] = GJ_L
        EIz = E * Iz
        k1 = 12.0 * EIz / L ** 3
        k2 = 6.0 * EIz / L ** 2
        k3 = 4.0 * EIz / L
        k4 = 2.0 * EIz / L
        Kloc[1, 1] = k1
        Kloc[1, 5] = k2
        Kloc[1, 7] = -k1
        Kloc[1, 11] = k2
        Kloc[5, 1] = k2
        Kloc[5, 5] = k3
        Kloc[5, 7] = -k2
        Kloc[5, 11] = k4
        Kloc[7, 1] = -k1
        Kloc[7, 5] = -k2
        Kloc[7, 7] = k1
        Kloc[7, 11] = -k2
        Kloc[11, 1] = k2
        Kloc[11, 5] = k4
        Kloc[11, 7] = -k2
        Kloc[11, 11] = k3
        EIy = E * Iy
        k5 = 12.0 * EIy / L ** 3
        k6 = 6.0 * EIy / L ** 2
        k7 = 4.0 * EIy / L
        k8 = 2.0 * EIy / L
        Kloc[2, 2] = k5
        Kloc[2, 4] = -k6
        Kloc[2, 8] = -k5
        Kloc[2, 10] = -k6
        Kloc[4, 2] = -k6
        Kloc[4, 4] = k7
        Kloc[4, 8] = k6
        Kloc[4, 10] = k8
        Kloc[8, 2] = -k5
        Kloc[8, 4] = k6
        Kloc[8, 8] = k5
        Kloc[8, 10] = k6
        Kloc[10, 2] = -k6
        Kloc[10, 4] = k8
        Kloc[10, 8] = k6
        Kloc[10, 10] = k7
        Lblk = np.zeros((12, 12), dtype=float)
        for b in range(4):
            Lblk[3 * b:3 * b + 3, 3 * b:3 * b + 3] = Lambda
        Ke = Lblk.T @ Kloc @ Lblk
        dofs_i = np.arange(6 * ni, 6 * ni + 6, dtype=int)
        dofs_j = np.arange(6 * nj, 6 * nj + 6, dtype=int)
        edofs = np.concatenate([dofs_i, dofs_j])
        K[np.ix_(edofs, edofs)] += Ke
    fixed = np.zeros(ndof, dtype=bool)
    if boundary_conditions is not None:
        for (n, bc) in boundary_conditions.items():
            idx0 = 6 * int(n)
            b = np.asarray(bc, dtype=int).reshape(-1)
            if b.size != 6:
                raise ValueError('Each boundary condition must have 6 entries (0 or 1).')
            fixed[idx0:idx0 + 6] = b.astype(bool)
    free = ~fixed
    u = np.zeros(ndof, dtype=float)
    if np.any(free):
        K_ff = K[np.ix_(free, free)]
        P_f = P[free]
        cond = np.linalg.cond(K_ff)
        if not np.isfinite(cond) or cond >= 1e+16:
            raise ValueError('Ill-conditioned free-free stiffness submatrix.')
        u_f = np.linalg.solve(K_ff, P_f)
        u[free] = u_f
    r = K @ u - P
    r[free] = 0.0
    return (u, r)