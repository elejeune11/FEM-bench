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
    import numpy as np
    from typing import Sequence
    import pytest
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an (N,3) array')
    N = node_coords.shape[0]
    ndof = 6 * N
    K = np.zeros((ndof, ndof), dtype=float)
    P = np.zeros(ndof, dtype=float)

    def node_dofs(n):
        base = 6 * n
        return np.array([base + i for i in range(6)], dtype=int)
    eps = 1e-12
    for el in elements:
        ni = int(el['node_i'])
        nj = int(el['node_j'])
        if not (0 <= ni < N and 0 <= nj < N):
            raise ValueError('Element node indices out of range')
        xi = node_coords[ni]
        xj = node_coords[nj]
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not L > eps:
            raise ValueError('Element length must be positive and non-zero')
        E = float(el['E'])
        nu = float(el['nu'])
        A = float(el['A'])
        Iy = float(el['I_y'])
        Iz = float(el['I_z'])
        J = float(el['J'])
        G = E / (2.0 * (1.0 + nu))
        ex = dx / L
        zref = el.get('local_z', None)
        if zref is None:
            vref = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(ex, vref)) > 1.0 - 1e-08:
                vref = np.array([0.0, 1.0, 0.0])
            zref = vref
        zref = np.asarray(zref, dtype=float).reshape(3)
        zref_proj = zref - np.dot(zref, ex) * ex
        nz = np.linalg.norm(zref_proj)
        if nz <= eps:
            raise ValueError('Provided local_z is parallel to the element axis')
        ez = zref_proj / nz
        ey = np.cross(ez, ex)
        Rrows = np.vstack((ex, ey, ez))
        Kl = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        Kl[0, 0] += k_ax
        Kl[0, 6] += -k_ax
        Kl[6, 0] += -k_ax
        Kl[6, 6] += k_ax
        k_tor = G * J / L
        Kl[3, 3] += k_tor
        Kl[3, 9] += -k_tor
        Kl[9, 3] += -k_tor
        Kl[9, 9] += k_tor
        facz = E * Iz / L ** 3
        (i_v, i_tz, j_v, j_tz) = (1, 5, 7, 11)
        m = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L]]) * facz
        idx = [i_v, i_tz, j_v, j_tz]
        for a in range(4):
            for b in range(4):
                Kl[idx[a], idx[b]] += m[a, b]
        facy = E * Iy / L ** 3
        (i_w, i_ty, j_w, j_ty) = (2, 4, 8, 10)
        m2 = np.array([[12.0, -6.0 * L, -12.0, -6.0 * L], [-6.0 * L, 4.0 * L * L, 6.0 * L, 2.0 * L * L], [-12.0, 6.0 * L, 12.0, 6.0 * L], [-6.0 * L, 2.0 * L * L, 6.0 * L, 4.0 * L * L]]) * facy
        idx2 = [i_w, i_ty, j_w, j_ty]
        for a in range(4):
            for b in range(4):
                Kl[idx2[a], idx2[b]] += m2[a, b]
        Tgl = np.zeros((12, 12), dtype=float)
        for blk in range(4):
            Tgl[blk * 3:(blk + 1) * 3, blk * 3:(blk + 1) * 3] = Rrows
        Ke = Tgl.T @ Kl @ Tgl
        dofs_i = node_dofs(ni)
        dofs_j = node_dofs(nj)
        edofs = np.concatenate((dofs_i, dofs_j))
        K[np.ix_(edofs, edofs)] += Ke
    for (n, loads) in (nodal_loads or {}).items():
        n = int(n)
        if not 0 <= n < N:
            raise ValueError('nodal_loads contains node index out of range')
        arr = np.asarray(loads, dtype=float).reshape(6)
        P[node_dofs(n)] += arr
    bc = np.zeros((N, 6), dtype=int)
    if boundary_conditions:
        for (n, flags) in boundary_conditions.items():
            n = int(n)
            if not 0 <= n < N:
                raise ValueError('boundary_conditions contains node index out of range')
            flags_arr = np.asarray(flags, dtype=int).reshape(6)
            bc[n, :] = flags_arr
    fixed_mask = bc.reshape(-1).astype(bool)
    free_mask = ~fixed_mask
    u = np.zeros(ndof, dtype=float)
    r = np.zeros(ndof, dtype=float)
    free_idx = np.nonzero(free_mask)[0]
    fixed_idx = np.nonzero(fixed_mask)[0]
    if free_idx.size > 0:
        K_ff = K[np.ix_(free_idx, free_idx)]
        P_f = P[free_idx]
        cond = np.linalg.cond(K_ff)
        if not np.isfinite(cond) or cond >= 1e+16:
            raise ValueError('Ill-conditioned free-free stiffness submatrix (cond >= 1e16)')
        u_f = np.linalg.solve(K_ff, P_f)
        u[free_idx] = u_f
    r_all = K @ u - P
    r[fixed_idx] = r_all[fixed_idx]
    r[free_idx] = 0.0
    return (u, r)