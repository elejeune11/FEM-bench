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
    N = node_coords.shape[0]
    ndof = 6 * N
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        Iy = elem['I_y']
        Iz = elem['I_z']
        J = elem['J']
        local_z_input = elem.get('local_z', None)
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if L == 0:
            raise ValueError('Zero-length element detected.')
        ex = np.array([dx, dy, dz]) / L
        if local_z_input is not None:
            ez = np.array(local_z_input)
            if np.allclose(np.cross(ex, ez), 0):
                raise ValueError('local_z must not be parallel to the beam axis.')
        elif abs(ex[2]) < 0.9:
            ez = np.array([0, 0, 1])
        else:
            ez = np.array([0, 1, 0])
        ey = np.cross(ez, ex)
        ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        T = np.zeros((12, 12))
        R = np.array([ex, ey, ez]).T
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        k_local = np.zeros((12, 12))
        EAL = E * A / L
        EIy_L3 = 12 * E * Iy / L ** 3
        EIy_L2 = 6 * E * Iy / L ** 2
        EIz_L3 = 12 * E * Iz / L ** 3
        EIz_L2 = 6 * E * Iz / L ** 2
        G = E / (2 * (1 + nu))
        GJ_L = G * J / L
        EIy_L = E * Iy / L
        EIz_L = E * Iz / L
        k_local[0, 0] = EAL
        k_local[6, 0] = -EAL
        k_local[6, 6] = EAL
        k_local[1, 1] = EIz_L3
        k_local[5, 1] = EIz_L2
        k_local[7, 1] = -EIz_L3
        k_local[11, 1] = EIz_L2
        k_local[5, 5] = 4 * EIz_L
        k_local[11, 5] = 2 * EIz_L
        k_local[7, 7] = EIz_L3
        k_local[11, 7] = -EIz_L2
        k_local[11, 11] = 4 * EIz_L
        k_local[2, 2] = EIy_L3
        k_local[4, 2] = -EIy_L2
        k_local[8, 2] = -EIy_L3
        k_local[10, 2] = -EIy_L2
        k_local[4, 4] = 4 * EIy_L
        k_local[10, 4] = 2 * EIy_L
        k_local[8, 8] = EIy_L3
        k_local[10, 8] = EIy_L2
        k_local[10, 10] = 4 * EIy_L
        k_local[3, 3] = GJ_L
        k_local[9, 3] = -GJ_L
        k_local[9, 9] = GJ_L
        k_local += k_local.T - np.diag(np.diag(k_local))
        k_global = T.T @ k_local @ T
        dof_indices = np.concatenate([np.arange(6 * node_i, 6 * node_i + 6), np.arange(6 * node_j, 6 * node_j + 6)])
        for (ii, idx) in enumerate(dof_indices):
            for (jj, jdx) in enumerate(dof_indices):
                K[idx, jdx] += k_global[ii, jj]
    for (node, load) in nodal_loads.items():
        F[6 * node:6 * node + 6] += np.array(load)
    is_fixed = np.zeros(ndof, dtype=bool)
    for (node, bc) in boundary_conditions.items():
        for i in range(6):
            if bc[i] == 1:
                is_fixed[6 * node + i] = True
    free_dofs = ~is_fixed
    fixed_dofs = is_fixed
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_fs = K[np.ix_(free_dofs, fixed_dofs)]
    K_sf = K[np.ix_(fixed_dofs, free_dofs)]
    K_ss = K[np.ix_(fixed_dofs, fixed_dofs)]
    F_f = F[free_dofs]
    F_s = F[fixed_dofs]
    cond_num = np.linalg.cond(K_ff)
    if cond_num >= 1e+16:
        raise ValueError('Free-free stiffness matrix is ill-conditioned.')
    u_f = np.linalg.solve(K_ff, F_f)
    u = np.zeros(ndof)
    u[free_dofs] = u_f
    r_s = K_sf @ u_f - F_s
    r = np.zeros(ndof)
    r[fixed_dofs] = r_s
    return (u, r)