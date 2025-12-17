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
    coords = np.asarray(node_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('node_coords must be an (N, 3) array.')
    N = coords.shape[0]
    ndof = 6 * N
    K = np.zeros((ndof, ndof), dtype=float)
    f = np.zeros(ndof, dtype=float)
    if nodal_loads is not None:
        for n, loads in nodal_loads.items():
            if n < 0 or n >= N:
                raise ValueError('nodal_loads contains invalid node index.')
            Lvec = np.asarray(loads, dtype=float).reshape(-1)
            if Lvec.size != 6:
                raise ValueError('Each nodal load must be a 6-element sequence.')
            idx = slice(6 * n, 6 * n + 6)
            f[idx] += Lvec

    def _element_rotation(xi: np.ndarray, xj: np.ndarray, local_z) -> np.ndarray:
        x_vec = xj - xi
        L = np.linalg.norm(x_vec)
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element has zero or invalid length.')
        x_dir = x_vec / L
        eps = 1e-12
        if local_z is None:
            z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if np.linalg.norm(np.cross(x_dir, z_ref)) < 1e-08:
                z_ref = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            z_ref = np.asarray(local_z, dtype=float).reshape(-1)
            if z_ref.size != 3:
                raise ValueError('local_z must be a 3-element vector.')
            nrm = np.linalg.norm(z_ref)
            if nrm <= eps:
                z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
                if np.linalg.norm(np.cross(x_dir, z_ref)) < 1e-08:
                    z_ref = np.array([0.0, 1.0, 0.0], dtype=float)
            else:
                z_ref = z_ref / nrm
        y_dir = np.cross(z_ref, x_dir)
        y_nrm = np.linalg.norm(y_dir)
        if y_nrm < 1e-10:
            alt = np.array([0.0, 1.0, 0.0], dtype=float)
            if np.linalg.norm(np.cross(x_dir, alt)) < 1e-08:
                alt = np.array([1.0, 0.0, 0.0], dtype=float)
            y_dir = np.cross(alt, x_dir)
            y_nrm = np.linalg.norm(y_dir)
            if y_nrm < 1e-12:
                raise ValueError('Cannot determine a valid local coordinate system for element.')
        y_dir = y_dir / y_nrm
        z_dir = np.cross(x_dir, y_dir)
        z_dir = z_dir / np.linalg.norm(z_dir)
        R = np.vstack((x_dir, y_dir, z_dir))
        return (R, L)

    def _beam_local_stiffness(E, nu, A, Iy, Iz, J, L) -> np.ndarray:
        G = E / (2.0 * (1.0 + nu))
        k = np.zeros((12, 12), dtype=float)
        EA_L = E * A / L
        k[0, 0] += EA_L
        k[0, 6] += -EA_L
        k[6, 0] += -EA_L
        k[6, 6] += EA_L
        GJ_L = G * J / L
        k[3, 3] += GJ_L
        k[3, 9] += -GJ_L
        k[9, 3] += -GJ_L
        k[9, 9] += GJ_L
        EI_z = E * Iz
        L2 = L * L
        L3 = L2 * L
        c1 = 12.0 * EI_z / L3
        c2 = 6.0 * EI_z / L2
        c3 = 4.0 * EI_z / L
        c4 = 2.0 * EI_z / L
        idx_vz = [1, 5, 7, 11]
        kbz = np.array([[c1, c2, -c1, c2], [c2, c3, -c2, c4], [-c1, -c2, c1, -c2], [c2, c4, -c2, c3]], dtype=float)
        for a in range(4):
            for b in range(4):
                k[idx_vz[a], idx_vz[b]] += kbz[a, b]
        EI_y = E * Iy
        c1y = 12.0 * EI_y / L3
        c2y = 6.0 * EI_y / L2
        c3y = 4.0 * EI_y / L
        c4y = 2.0 * EI_y / L
        idx_y = [2, 4, 8, 10]
        kby = np.array([[c1y, -c2y, -c1y, -c2y], [-c2y, c3y, c2y, c4y], [-c1y, c2y, c1y, c2y], [-c2y, c4y, c2y, c3y]], dtype=float)
        for a in range(4):
            for b in range(4):
                k[idx_y[a], idx_y[b]] += kby[a, b]
        return k
    for e in elements:
        ni = int(e['node_i'])
        nj = int(e['node_j'])
        if ni < 0 or nj < 0 or ni >= N or (nj >= N):
            raise ValueError('Element node indices out of range.')
        xi = coords[ni]
        xj = coords[nj]
        local_z = e.get('local_z', None)
        R, L = _element_rotation(xi, xj, local_z)
        E = float(e['E'])
        nu = float(e['nu'])
        A = float(e['A'])
        Iy = float(e['I_y'])
        Iz = float(e['I_z'])
        J = float(e['J'])
        k_loc = _beam_local_stiffness(E, nu, A, Iy, Iz, J, L)
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        k_glob = T.T @ k_loc @ T
        dof_map = np.r_[np.arange(6 * ni, 6 * ni + 6), np.arange(6 * nj, 6 * nj + 6)]
        for a_local in range(12):
            Aglob = dof_map[a_local]
            K[Aglob, dof_map] += k_glob[a_local, :]
    fixed = np.zeros(ndof, dtype=bool)
    if boundary_conditions is not None:
        for n, bc in boundary_conditions.items():
            if n < 0 or n >= N:
                raise ValueError('boundary_conditions contains invalid node index.')
            bc_arr = np.asarray(bc, dtype=int).reshape(-1)
            if bc_arr.size != 6:
                raise ValueError('Each boundary condition must be a 6-element sequence.')
            bi = 6 * n
            for i in range(6):
                fixed[bi + i] = bool(bc_arr[i])
    free = ~fixed
    u = np.zeros(ndof, dtype=float)
    r = np.zeros(ndof, dtype=float)
    n_free = int(np.sum(free))
    if n_free > 0:
        Kff = K[np.ix_(free, free)]
        ff = f[free]
        try:
            cond = np.linalg.cond(Kff)
        except np.linalg.LinAlgError:
            cond = np.inf
        if not np.isfinite(cond) or cond >= 1e+16:
            raise ValueError('The free-free stiffness matrix is ill-conditioned or singular; analysis aborted.')
        u_free = np.linalg.solve(Kff, ff)
        u[free] = u_free
    Ku = K @ u
    r[fixed] = Ku[fixed] - f[fixed]
    return (u, r)