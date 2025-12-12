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
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be (N,3) array')
    N = node_coords.shape[0]
    Ndof = 6 * N
    K = np.zeros((Ndof, Ndof), dtype=float)
    F = np.zeros((Ndof,), dtype=float)
    for (n, load) in nodal_loads.items():
        if load is None:
            continue
        load_arr = np.asarray(load, dtype=float)
        if load_arr.size != 6:
            raise ValueError('Each nodal load must be length 6')
        idx = 6 * int(n)
        F[idx:idx + 6] += load_arr
    for el in elements:
        ni = int(el['node_i'])
        nj = int(el['node_j'])
        xi = node_coords[ni]
        xj = node_coords[nj]
        dx = xj - xi
        L = np.linalg.norm(dx)
        if L <= 0:
            raise ValueError('Element with zero length')
        ex = dx / L
        local_z = el.get('local_z', None)
        if local_z is None:
            g_z = np.array([0.0, 0.0, 1.0], dtype=float)
            g_y = np.array([0.0, 1.0, 0.0], dtype=float)
            if abs(np.dot(g_z, ex)) > 0.999999:
                ez = g_y.copy()
            else:
                ez = g_z.copy()
        else:
            ez = np.asarray(local_z, dtype=float)
            if ez.shape != (3,):
                raise ValueError('local_z must be length 3')
            nrm = np.linalg.norm(ez)
            if nrm < 1e-12:
                raise ValueError('local_z must be nonzero')
            ez = ez / nrm
            if abs(np.dot(ez, ex)) > 0.999999:
                g_z = np.array([0.0, 0.0, 1.0], dtype=float)
                g_y = np.array([0.0, 1.0, 0.0], dtype=float)
                if abs(np.dot(g_z, ex)) > 0.999999:
                    ez = g_y
                else:
                    ez = g_z
        ey_temp = np.cross(ez, ex)
        ey_nrm = np.linalg.norm(ey_temp)
        if ey_nrm < 1e-12:
            if abs(ex[0]) < abs(ex[1]):
                tmp = np.array([1.0, 0.0, 0.0])
            else:
                tmp = np.array([0.0, 1.0, 0.0])
            ey_temp = np.cross((temp := tmp), ex)
            ey_nrm = np.linalg.norm(ey_temp)
            if ey_nrm < 1e-12:
                ey_temp = np.cross(np.array([0.0, 0.0, 1.0]), ex)
                ey_nrm = np.linalg.norm(ey_temp)
        ey = ey_temp / ey_nrm
        ez = np.cross(ex, ey)
        ez = ez / np.linalg.norm(ez)
        R = np.vstack((ex, ey, ez)).astype(float)
        E = float(el['E'])
        nu = float(el['nu'])
        A = float(el['A'])
        I_y = float(el['I_y'])
        I_z = float(el['I_z'])
        J = float(el['J'])
        G = E / (2.0 * (1.0 + nu))
        k = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        k[0, 0] = k_ax
        k[0, 6] = -k_ax
        k[6, 0] = -k_ax
        k[6, 6] = k_ax
        k_t = G * J / L
        k[3, 3] = k_t
        k[3, 9] = -k_t
        k[9, 3] = -k_t
        k[9, 9] = k_t
        EI_z = E * I_z
        coeff_z = EI_z / L ** 3
        k_z = coeff_z * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L]], dtype=float)
        inds_z = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                k[inds_z[a], inds_z[b]] += k_z[a, b]
        EI_y = E * I_y
        coeff_y = EI_y / L ** 3
        k_y = coeff_y * np.array([[12.0, -6.0 * L, -12.0, -6.0 * L], [-6.0 * L, 4.0 * L * L, 6.0 * L, 2.0 * L * L], [-12.0, 6.0 * L, 12.0, 6.0 * L], [-6.0 * L, 2.0 * L * L, 6.0 * L, 4.0 * L * L]], dtype=float)
        inds_y = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                k[inds_y[a], inds_y[b]] += k_y[a, b]
        T = np.zeros((12, 12), dtype=float)
        for i_block in range(4):
            T[3 * i_block:3 * i_block + 3, 3 * i_block:3 * i_block + 3] = R
        k_global = T.T @ k @ T
        dofs_i = np.arange(6 * ni, 6 * ni + 6)
        dofs_j = np.arange(6 * nj, 6 * nj + 6)
        dofs = np.concatenate((dofs_i, dofs_j))
        for (a_local, A_glob) in enumerate(dofs):
            for (b_local, B_glob) in enumerate(dofs):
                K[A_glob, B_glob] += k_global[a_local, b_local]
    fixed = np.zeros((Ndof,), dtype=bool)
    for (node_idx, bc) in boundary_conditions.items():
        arr = np.asarray(bc, dtype=int)
        if arr.size != 6:
            raise ValueError('Each boundary condition entry must have length 6')
        idx = 6 * int(node_idx)
        for i in range(6):
            if int(arr[i]) != 0:
                fixed[idx + i] = True
    free_dofs = np.where(~fixed)[0]
    fixed_dofs = np.where(fixed)[0]
    u = np.zeros((Ndof,), dtype=float)
    if free_dofs.size > 0:
        Kff = K[np.ix_(free_dofs, free_dofs)]
        Ff = F[free_dofs]
        try:
            cond = np.linalg.cond(Kff)
        except Exception:
            cond = np.inf
        if not np.isfinite(cond) or cond >= 1e+16:
            raise ValueError('Free-free stiffness matrix is ill-conditioned or singular (cond=%s)' % repr(cond))
        u_f = np.linalg.solve(Kff, Ff)
        u[free_dofs] = u_f
    else:
        u[:] = 0.0
    r = K.dot(u) - F
    r_out = np.zeros_like(r)
    r_out[fixed_dofs] = r[fixed_dofs]
    u_out = np.zeros_like(u)
    u_out[free_dofs] = u[free_dofs]
    return (u_out, r_out)