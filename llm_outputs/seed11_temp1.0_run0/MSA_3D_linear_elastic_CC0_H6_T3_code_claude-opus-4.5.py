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
    K_global = np.zeros((ndof, ndof))
    F_global = np.zeros(ndof)
    for (node_idx, loads) in nodal_loads.items():
        dofs = np.arange(6 * node_idx, 6 * node_idx + 6)
        F_global[dofs] += np.array(loads)
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        local_z = elem.get('local_z', None)
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        dx = xj - xi
        dy = yj - yi
        dz = zj - zi
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        x_axis = np.array([dx, dy, dz]) / L
        tol = 1e-06
        if local_z is None:
            if abs(abs(x_axis[2]) - 1.0) < tol:
                local_z = np.array([0.0, 1.0, 0.0])
            else:
                local_z = np.array([0.0, 0.0, 1.0])
        else:
            local_z = np.array(local_z, dtype=float)
            local_z = local_z / np.linalg.norm(local_z)
        y_axis = np.cross(local_z, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        T_small = np.array([x_axis, y_axis, z_axis])
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = T_small
        G = E / (2 * (1 + nu))
        k_local = np.zeros((12, 12))
        EA_L = E * A / L
        k_local[0, 0] = EA_L
        k_local[0, 6] = -EA_L
        k_local[6, 0] = -EA_L
        k_local[6, 6] = EA_L
        GJ_L = G * J / L
        k_local[3, 3] = GJ_L
        k_local[3, 9] = -GJ_L
        k_local[9, 3] = -GJ_L
        k_local[9, 9] = GJ_L
        EIz_L3 = E * I_z / L ** 3
        k_local[1, 1] = 12 * EIz_L3
        k_local[1, 5] = 6 * EIz_L3 * L
        k_local[1, 7] = -12 * EIz_L3
        k_local[1, 11] = 6 * EIz_L3 * L
        k_local[5, 1] = 6 * EIz_L3 * L
        k_local[5, 5] = 4 * EIz_L3 * L ** 2
        k_local[5, 7] = -6 * EIz_L3 * L
        k_local[5, 11] = 2 * EIz_L3 * L ** 2
        k_local[7, 1] = -12 * EIz_L3
        k_local[7, 5] = -6 * EIz_L3 * L
        k_local[7, 7] = 12 * EIz_L3
        k_local[7, 11] = -6 * EIz_L3 * L
        k_local[11, 1] = 6 * EIz_L3 * L
        k_local[11, 5] = 2 * EIz_L3 * L ** 2
        k_local[11, 7] = -6 * EIz_L3 * L
        k_local[11, 11] = 4 * EIz_L3 * L ** 2
        EIy_L3 = E * I_y / L ** 3
        k_local[2, 2] = 12 * EIy_L3
        k_local[2, 4] = -6 * EIy_L3 * L
        k_local[2, 8] = -12 * EIy_L3
        k_local[2, 10] = -6 * EIy_L3 * L
        k_local[4, 2] = -6 * EIy_L3 * L
        k_local[4, 4] = 4 * EIy_L3 * L ** 2
        k_local[4, 8] = 6 * EIy_L3 * L
        k_local[4, 10] = 2 * EIy_L3 * L ** 2
        k_local[8, 2] = -12 * EIy_L3
        k_local[8, 4] = 6 * EIy_L3 * L
        k_local[8, 8] = 12 * EIy_L3
        k_local[8, 10] = 6 * EIy_L3 * L
        k_local[10, 2] = -6 * EIy_L3 * L
        k_local[10, 4] = 2 * EIy_L3 * L ** 2
        k_local[10, 8] = 6 * EIy_L3 * L
        k_local[10, 10] = 4 * EIy_L3 * L ** 2
        k_global_elem = T.T @ k_local @ T
        dofs_i = np.arange(6 * node_i, 6 * node_i + 6)
        dofs_j = np.arange(6 * node_j, 6 * node_j + 6)
        elem_dofs = np.concatenate([dofs_i, dofs_j])
        for (ii, di) in enumerate(elem_dofs):
            for (jj, dj) in enumerate(elem_dofs):
                K_global[di, dj] += k_global_elem[ii, jj]
    fixed_dofs = []
    free_dofs = []
    for node_idx in range(N):
        bc = boundary_conditions.get(node_idx, [0, 0, 0, 0, 0, 0])
        for local_dof in range(6):
            global_dof = 6 * node_idx + local_dof
            if bc[local_dof] == 1:
                fixed_dofs.append(global_dof)
            else:
                free_dofs.append(global_dof)
    free_dofs = np.array(free_dofs, dtype=int)
    fixed_dofs = np.array(fixed_dofs, dtype=int)
    u = np.zeros(ndof)
    r = np.zeros(ndof)
    if len(free_dofs) > 0:
        K_ff = K_global[np.ix_(free_dofs, free_dofs)]
        F_f = F_global[free_dofs]
        cond_num = np.linalg.cond(K_ff)
        if cond_num >= 1e+16:
            raise ValueError('The free-free stiffness matrix is ill-conditioned and the system cannot be reliably solved.')
        u_f = np.linalg.solve(K_ff, F_f)
        u[free_dofs] = u_f
    if len(fixed_dofs) > 0:
        r[fixed_dofs] = K_global[np.ix_(fixed_dofs, free_dofs)] @ u[free_dofs] - F_global[fixed_dofs] if len(free_dofs) > 0 else -F_global[fixed_dofs]
    return (u, r)