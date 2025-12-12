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
        dof_start = 6 * node_idx
        F_global[dof_start:dof_start + 6] += np.array(loads)
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
        coord_i = node_coords[node_i]
        coord_j = node_coords[node_j]
        delta = coord_j - coord_i
        L = np.linalg.norm(delta)
        x_axis = delta / L
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
        R = np.array([x_axis, y_axis, z_axis])
        G = E / (2 * (1 + nu))
        k_local = np.zeros((12, 12))
        k_local[0, 0] = E * A / L
        k_local[0, 6] = -E * A / L
        k_local[6, 0] = -E * A / L
        k_local[6, 6] = E * A / L
        k_local[3, 3] = G * J / L
        k_local[3, 9] = -G * J / L
        k_local[9, 3] = -G * J / L
        k_local[9, 9] = G * J / L
        k1 = 12 * E * I_z / L ** 3
        k2 = 6 * E * I_z / L ** 2
        k3 = 4 * E * I_z / L
        k4 = 2 * E * I_z / L
        k_local[1, 1] = k1
        k_local[1, 5] = k2
        k_local[1, 7] = -k1
        k_local[1, 11] = k2
        k_local[5, 1] = k2
        k_local[5, 5] = k3
        k_local[5, 7] = -k2
        k_local[5, 11] = k4
        k_local[7, 1] = -k1
        k_local[7, 5] = -k2
        k_local[7, 7] = k1
        k_local[7, 11] = -k2
        k_local[11, 1] = k2
        k_local[11, 5] = k4
        k_local[11, 7] = -k2
        k_local[11, 11] = k3
        k5 = 12 * E * I_y / L ** 3
        k6 = 6 * E * I_y / L ** 2
        k7 = 4 * E * I_y / L
        k8 = 2 * E * I_y / L
        k_local[2, 2] = k5
        k_local[2, 4] = -k6
        k_local[2, 8] = -k5
        k_local[2, 10] = -k6
        k_local[4, 2] = -k6
        k_local[4, 4] = k7
        k_local[4, 8] = k6
        k_local[4, 10] = k8
        k_local[8, 2] = -k5
        k_local[8, 4] = k6
        k_local[8, 8] = k5
        k_local[8, 10] = k6
        k_local[10, 2] = -k6
        k_local[10, 4] = k8
        k_local[10, 8] = k6
        k_local[10, 10] = k7
        T = np.zeros((12, 12))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        k_global_elem = T.T @ k_local @ T
        dofs_i = np.arange(6 * node_i, 6 * node_i + 6)
        dofs_j = np.arange(6 * node_j, 6 * node_j + 6)
        dofs = np.concatenate([dofs_i, dofs_j])
        for ii in range(12):
            for jj in range(12):
                K_global[dofs[ii], dofs[jj]] += k_global_elem[ii, jj]
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
        cond = np.linalg.cond(K_ff)
        if cond >= 1e+16:
            raise ValueError('The free-free stiffness matrix is ill-conditioned and the system cannot be reliably solved.')
        u_f = np.linalg.solve(K_ff, F_f)
        u[free_dofs] = u_f
    if len(fixed_dofs) > 0:
        r[fixed_dofs] = K_global[np.ix_(fixed_dofs, free_dofs)] @ u[free_dofs] - F_global[fixed_dofs] if len(free_dofs) > 0 else -F_global[fixed_dofs]
    return (u, r)