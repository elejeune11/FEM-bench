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
    from typing import Sequence
    import pytest
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an (N, 3) array')
    N = node_coords.shape[0]
    ndof = 6 * N
    K = np.zeros((ndof, ndof), dtype=float)
    F = np.zeros(ndof, dtype=float)
    if nodal_loads is not None:
        for (n, loads) in nodal_loads.items():
            if n < 0 or n >= N:
                raise ValueError('Node index in nodal_loads out of range')
            loads_arr = np.asarray(loads, dtype=float)
            if loads_arr.size != 6:
                raise ValueError('Each nodal load must have 6 components')
            F[n * 6:n * 6 + 6] += loads_arr

    def local_triad(xi: np.ndarray, xj: np.ndarray, local_z_guess):
        dx = xj - xi
        L = np.linalg.norm(dx)
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element has zero or invalid length')
        ex = dx / L
        if local_z_guess is None:
            zg = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(zg, ex)) > 0.99:
                zg = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            zg = np.asarray(local_z_guess, dtype=float)
            nz = np.linalg.norm(zg)
            if nz <= 0.0:
                raise ValueError('Provided local_z has zero length')
            zg = zg / nz
            if abs(np.dot(zg, ex)) >= 1.0 - 1e-08:
                raise ValueError('Provided local_z is parallel to the element axis')
        zproj = zg - np.dot(zg, ex) * ex
        nzp = np.linalg.norm(zproj)
        if nzp < 1e-12:
            alt = np.array([0.0, 1.0, 0.0], dtype=float)
            if abs(np.dot(alt, ex)) > 0.99:
                alt = np.array([0.0, 0.0, 1.0], dtype=float)
            zproj = alt - np.dot(alt, ex) * ex
            nzp = np.linalg.norm(zproj)
            if nzp < 1e-12:
                raise ValueError('Cannot define a valid local z-axis for the element')
        ez = zproj / nzp
        ey = np.cross(ez, ex)
        ny = np.linalg.norm(ey)
        if ny < 1e-12:
            raise ValueError('Failed to construct orthonormal local triad')
        ey = ey / ny
        ez = np.cross(ex, ey)
        return (ex, ey, ez, L)
    for el in elements:
        i = int(el['node_i'])
        j = int(el['node_j'])
        if i < 0 or i >= N or j < 0 or (j >= N):
            raise ValueError('Element node index out of range')
        xi = node_coords[i]
        xj = node_coords[j]
        local_z_guess = el.get('local_z', None)
        (ex, ey, ez, L) = local_triad(xi, xj, local_z_guess)
        E = float(el['E'])
        nu = float(el['nu'])
        A = float(el['A'])
        Iy = float(el['I_y'])
        Iz = float(el['I_z'])
        J = float(el['J'])
        if L <= 0.0:
            raise ValueError('Element length must be positive')
        G = E / (2.0 * (1.0 + nu))
        k = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        k[0, 0] += k_ax
        k[0, 6] -= k_ax
        k[6, 0] -= k_ax
        k[6, 6] += k_ax
        k_to = G * J / L
        k[3, 3] += k_to
        k[3, 9] -= k_to
        k[9, 3] -= k_to
        k[9, 9] += k_to
        EIz = E * Iz
        kvv = 12.0 * EIz / L ** 3
        kvr = 6.0 * EIz / L ** 2
        krr1 = 4.0 * EIz / L
        krr2 = 2.0 * EIz / L
        (v1, rz1, v2, rz2) = (1, 5, 7, 11)
        k[v1, v1] += kvv
        k[v1, rz1] += kvr
        k[v1, v2] -= kvv
        k[v1, rz2] += kvr
        k[rz1, v1] += kvr
        k[rz1, rz1] += krr1
        k[rz1, v2] -= kvr
        k[rz1, rz2] += krr2
        k[v2, v1] -= kvv
        k[v2, rz1] -= kvr
        k[v2, v2] += kvv
        k[v2, rz2] -= kvr
        k[rz2, v1] += kvr
        k[rz2, rz1] += krr2
        k[rz2, v2] -= kvr
        k[rz2, rz2] += krr1
        EIy = E * Iy
        kvv = 12.0 * EIy / L ** 3
        kvr = 6.0 * EIy / L ** 2
        krr1 = 4.0 * EIy / L
        krr2 = 2.0 * EIy / L
        (w1, ry1, w2, ry2) = (2, 4, 8, 10)
        k[w1, w1] += kvv
        k[w1, ry1] += kvr
        k[w1, w2] -= kvv
        k[w1, ry2] += kvr
        k[ry1, w1] += kvr
        k[ry1, ry1] += krr1
        k[ry1, w2] -= kvr
        k[ry1, ry2] += krr2
        k[w2, w1] -= kvv
        k[w2, ry1] -= kvr
        k[w2, w2] += kvv
        k[w2, ry2] -= kvr
        k[ry2, w1] += kvr
        k[ry2, ry1] += krr2
        k[ry2, w2] -= kvr
        k[ry2, ry2] += krr1
        Q = np.column_stack((ex, ey, ez))
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = Q
        T[3:6, 3:6] = Q
        T[6:9, 6:9] = Q
        T[9:12, 9:12] = Q
        kg = T @ k @ T.T
        idx_i = np.arange(i * 6, i * 6 + 6)
        idx_j = np.arange(j * 6, j * 6 + 6)
        idx = np.concatenate((idx_i, idx_j))
        K[np.ix_(idx, idx)] += kg
    fixed = np.zeros(ndof, dtype=bool)
    if boundary_conditions is not None:
        for (n, bc) in boundary_conditions.items():
            if n < 0 or n >= N:
                raise ValueError('Node index in boundary_conditions out of range')
            bc_arr = np.asarray(bc, dtype=int)
            if bc_arr.size != 6:
                raise ValueError('Each boundary condition must have 6 components')
            fixed[n * 6:n * 6 + 6] = bc_arr.astype(bool)
    free = np.where(~fixed)[0]
    u = np.zeros(ndof, dtype=float)
    if free.size > 0:
        K_ff = K[np.ix_(free, free)]
        F_f = F[free]
        cond = np.linalg.cond(K_ff)
        if not np.isfinite(cond) or cond >= 1e+16:
            raise ValueError('Free-free stiffness matrix is ill-conditioned or singular')
        u_f = np.linalg.solve(K_ff, F_f)
        u[free] = u_f
    r = K @ u - F
    r[free] = 0.0
    return (u, r)