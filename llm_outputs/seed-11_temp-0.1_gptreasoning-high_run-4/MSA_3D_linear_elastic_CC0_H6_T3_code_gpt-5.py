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
    coords = np.asarray(node_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('node_coords must be an (N, 3) array of node coordinates.')
    N = coords.shape[0]
    ndof = 6 * N
    K = np.zeros((ndof, ndof), dtype=float)
    F = np.zeros(ndof, dtype=float)
    if nodal_loads is not None:
        for n, load in nodal_loads.items():
            if not 0 <= n < N:
                raise ValueError(f'Invalid node index in nodal_loads: {n}')
            arr = np.asarray(load, dtype=float).reshape(-1)
            if arr.size != 6:
                raise ValueError('Each nodal load must be a 6-element sequence [Fx, Fy, Fz, Mx, My, Mz].')
            F[6 * n:6 * n + 6] += arr

    def element_geometry(p_i: np.ndarray, p_j: np.ndarray, local_z_vec):
        ex_vec = p_j - p_i
        L = float(np.linalg.norm(ex_vec))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element has zero or invalid length.')
        ex = ex_vec / L
        if local_z_vec is None:
            zref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(ex, zref))) > 1.0 - 1e-08:
                zref = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            zref = np.asarray(local_z_vec, dtype=float).reshape(3)
            if not np.all(np.isfinite(zref)):
                zref = np.array([0.0, 0.0, 1.0], dtype=float)
                if abs(float(np.dot(ex, zref))) > 1.0 - 1e-08:
                    zref = np.array([0.0, 1.0, 0.0], dtype=float)
        v = zref - float(np.dot(zref, ex)) * ex
        nv = float(np.linalg.norm(v))
        if nv < 1e-12:
            zref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(ex, zref))) > 1.0 - 1e-08:
                zref = np.array([0.0, 1.0, 0.0], dtype=float)
            v = zref - float(np.dot(zref, ex)) * ex
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                axes = np.eye(3)
                cand = axes[int(np.argmin(np.abs(ex)))]
                v = cand - float(np.dot(cand, ex)) * ex
                nv = float(np.linalg.norm(v))
                if nv < 1e-14:
                    raise ValueError('Cannot construct a valid local coordinate system for the element.')
        ez_temp = v / nv
        ey = np.cross(ez_temp, ex)
        ney = float(np.linalg.norm(ey))
        if ney < 1e-12:
            axes = np.eye(3)
            cand = axes[int(np.argmax(np.abs(ex)))]
            v = cand - float(np.dot(cand, ex)) * ex
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                cand = np.array([1.0, 0.0, 0.0], dtype=float)
                if abs(float(np.dot(ex, cand))) > 1.0 - 1e-12:
                    cand = np.array([0.0, 1.0, 0.0], dtype=float)
                v = cand - float(np.dot(cand, ex)) * ex
                nv = float(np.linalg.norm(v))
                if nv < 1e-12:
                    raise ValueError('Cannot construct a valid local coordinate system for the element.')
            ez_temp = v / nv
            ey = np.cross(ez_temp, ex)
            ney = float(np.linalg.norm(ey))
            if ney < 1e-12:
                raise ValueError('Cannot construct a valid local coordinate system for the element.')
        ey = ey / ney
        ez = np.cross(ex, ey)
        ex = ex / float(np.linalg.norm(ex))
        ey = ey / float(np.linalg.norm(ey))
        ez = ez / float(np.linalg.norm(ez))
        R = np.vstack([ex, ey, ez])
        return (L, R)
    for e in elements:
        try:
            i = int(e['node_i'])
            j = int(e['node_j'])
            E = float(e['E'])
            nu = float(e['nu'])
            A = float(e['A'])
            I_y = float(e['I_y'])
            I_z = float(e['I_z'])
            J = float(e['J'])
            local_z = e.get('local_z', None)
        except Exception as exc:
            raise ValueError(f'Invalid element definition: {e}') from exc
        if not (0 <= i < N and 0 <= j < N):
            raise ValueError(f'Element node indices out of range: {i}, {j}')
        if i == j:
            raise ValueError(f'Element has identical end nodes: {i}')
        p_i = coords[i]
        p_j = coords[j]
        L, R = element_geometry(p_i, p_j, local_z)
        G = E / (2.0 * (1.0 + nu))
        k = np.zeros((12, 12), dtype=float)
        iUX, iUY, iUZ, iRX, iRY, iRZ = (0, 1, 2, 3, 4, 5)
        jUX, jUY, jUZ, jRX, jRY, jRZ = (6, 7, 8, 9, 10, 11)
        k_ax = E * A / L
        k[iUX, iUX] += k_ax
        k[iUX, jUX] -= k_ax
        k[jUX, iUX] -= k_ax
        k[jUX, jUX] += k_ax
        k_t = G * J / L
        k[iRX, iRX] += k_t
        k[iRX, jRX] -= k_t
        k[jRX, iRX] -= k_t
        k[jRX, jRX] += k_t
        EIz = E * I_z
        kz1 = 12.0 * EIz / L ** 3
        kz2 = 6.0 * EIz / L ** 2
        kz3 = 4.0 * EIz / L
        kz4 = 2.0 * EIz / L
        a, b, c, d = (iUY, iRZ, jUY, jRZ)
        k[a, a] += kz1
        k[a, b] += kz2
        k[a, c] += -kz1
        k[a, d] += kz2
        k[b, b] += kz3
        k[b, c] += -kz2
        k[b, d] += kz4
        k[c, c] += kz1
        k[c, d] += -kz2
        k[d, d] += kz3
        k[b, a] = k[a, b]
        k[c, a] = k[a, c]
        k[d, a] = k[a, d]
        k[c, b] = k[b, c]
        k[d, b] = k[b, d]
        k[d, c] = k[c, d]
        EIy = E * I_y
        ky1 = 12.0 * EIy / L ** 3
        ky2 = 6.0 * EIy / L ** 2
        ky3 = 4.0 * EIy / L
        ky4 = 2.0 * EIy / L
        a2, b2, c2, d2 = (iUZ, iRY, jUZ, jRY)
        k[a2, a2] += ky1
        k[a2, b2] += -ky2
        k[a2, c2] += -ky1
        k[a2, d2] += -ky2
        k[b2, b2] += ky3
        k[b2, c2] += ky2
        k[b2, d2] += ky4
        k[c2, c2] += ky1
        k[c2, d2] += ky2
        k[d2, d2] += ky3
        k[b2, a2] = k[a2, b2]
        k[c2, a2] = k[a2, c2]
        k[d2, a2] = k[a2, d2]
        k[c2, b2] = k[b2, c2]
        k[d2, b2] = k[b2, d2]
        k[d2, c2] = k[c2, d2]
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        k_global = T.T @ k @ T
        dofs = np.array([6 * i + 0, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * j + 0, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5], dtype=int)
        K[np.ix_(dofs, dofs)] += k_global
    fixed = np.zeros(ndof, dtype=bool)
    if boundary_conditions is not None:
        for n, bc in boundary_conditions.items():
            if not 0 <= n < N:
                raise ValueError(f'Invalid node index in boundary_conditions: {n}')
            vals = np.asarray(list(bc), dtype=int).reshape(-1)
            if vals.size != 6:
                raise ValueError('Each boundary condition must be a 6-element sequence of 0 (free) or 1 (fixed).')
            fixed[6 * n:6 * n + 6] = vals.astype(bool)
    free = ~fixed
    u = np.zeros(ndof, dtype=float)
    if np.any(free):
        K_ff = K[np.ix_(free, free)]
        F_f = F[free]
        try:
            cond_number = np.linalg.cond(K_ff)
        except np.linalg.LinAlgError as exc:
            raise ValueError('Free-free stiffness matrix is singular or ill-conditioned.') from exc
        if not np.isfinite(cond_number) or cond_number >= 1e+16:
            raise ValueError('Free-free stiffness matrix is ill-conditioned or singular (cond >= 1e16).')
        u_f = np.linalg.solve(K_ff, F_f)
        u[free] = u_f
    else:
        pass
    Ku = K @ u
    r = np.zeros(ndof, dtype=float)
    r[fixed] = Ku[fixed] - F[fixed]
    return (u, r)