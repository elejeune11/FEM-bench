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
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K_global = np.zeros((n_dofs, n_dofs))
    F_global = np.zeros(n_dofs)
    for (node_idx, loads) in nodal_loads.items():
        dof_start = 6 * node_idx
        F_global[dof_start:dof_start + 6] = loads
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        coord_i = node_coords[node_i]
        coord_j = node_coords[node_j]
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        L_vec = coord_j - coord_i
        L = np.linalg.norm(L_vec)
        x_local = L_vec / L
        if 'local_z' in elem and elem['local_z'] is not None:
            z_local = np.array(elem['local_z'])
            z_local = z_local / np.linalg.norm(z_local)
        elif abs(x_local[2]) < 0.9:
            z_local = np.array([0, 0, 1])
        else:
            z_local = np.array([0, 1, 0])
        z_local = z_local - np.dot(z_local, x_local) * x_local
        z_local = z_local / np.linalg.norm(z_local)
        y_local = np.cross(z_local, x_local)
        y_local = y_local / np.linalg.norm(y_local)
        T = np.zeros((6, 6))
        T[0:3, 0:3] = np.array([x_local, y_local, z_local])
        T[3:6, 3:6] = np.array([x_local, y_local, z_local])
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
        k_local[2, 2] = 12 * E * I_z / L ** 3
        k_local[2, 4] = 6 * E * I_z / L ** 2
        k_local[2, 8] = -12 * E * I_z / L ** 3
        k_local[2, 10] = 6 * E * I_z / L ** 2
        k_local[4, 2] = 6 * E * I_z / L ** 2
        k_local[4, 4] = 4 * E * I_z / L
        k_local[4, 8] = -6 * E * I_z / L ** 2
        k_local[4, 10] = 2 * E * I_z / L
        k_local[8, 2] = -12 * E * I_z / L ** 3
        k_local[8, 4] = -6 * E * I_z / L ** 2
        k_local[8, 8] = 12 * E * I_z / L ** 3
        k_local[8, 10] = -6 * E * I_z / L ** 2
        k_local[10, 2] = 6 * E * I_z / L ** 2
        k_local[10, 4] = 2 * E * I_z / L
        k_local[10, 8] = -6 * E * I_z / L ** 2
        k_local[10, 10] = 4 * E * I_z / L
        k_local[1, 1] = 12 * E * I_y / L ** 3
        k_local[1, 5] = -6 * E * I_y / L ** 2
        k_local[1, 7] = -12 * E * I_y / L ** 3
        k_local[1, 11] = -6 * E * I_y / L ** 2
        k_local[5, 1] = -6 * E * I_y / L ** 2
        k_local[5, 5] = 4 * E * I_y / L
        k_local[5, 7] = 6 * E * I_y / L ** 2
        k_local[5, 11] = 2 * E * I_y / L
        k_local[7, 1] = -12 * E * I_y / L ** 3
        k_local[7, 5] = 6 * E * I_y / L ** 2
        k_local[7, 7] = 12 * E * I_y / L ** 3
        k_local[7, 11] = 6 * E * I_y / L ** 2
        k_local[11, 1] = -6 * E * I_y / L ** 2
        k_local[11, 5] = 2 * E * I_y / L
        k_local[11, 7] = 6 * E * I_y / L ** 2
        k_local[11, 11] = 4 * E * I_y / L
        T_full = np.zeros((12, 12))
        T_full[0:6, 0:6] = T
        T_full[6:12, 6:12] = T
        k_global = T_full.T @ k_local @ T_full
        dofs = np.concatenate([np.arange(6 * node_i, 6 * node_i + 6), np.arange(6 * node_j, 6 * node_j + 6)])
        for (i, dof_i) in enumerate(dofs):
            for (j, dof_j) in enumerate(dofs):
                K_global[dof_i, dof_j] += k_global[i, j]
    fixed_dofs = []
    free_dofs = []
    for node_idx in range(n_nodes):
        if node_idx in boundary_conditions:
            bc = boundary_conditions[node_idx]
            for dof_local in range(6):
                dof_global = 6 * node_idx + dof_local
                if bc[dof_local] == 1:
                    fixed_dofs.append(dof_global)
                else:
                    free_dofs.append(dof_global)
        else:
            for dof_local in range(6):
                dof_global = 6 * node_idx + dof_local
                free_dofs.append(dof_global)
    free_dofs = np.array(free_dofs)
    fixed_dofs = np.array(fixed_dofs)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_fc = K_global[np.ix_(free_dofs, fixed_dofs)]
    K_cf = K_global[np.ix_(fixed_dofs, free_dofs)]
    K_cc = K_global[np.ix_(fixed_dofs, fixed_dofs)]
    F_f = F_global[free_dofs]
    F_c = F_global[fixed_dofs]
    cond_num = np.linalg.cond(K_ff)
    if cond_num >= 1e+16:
        raise ValueError(f'Free-free stiffness matrix is ill-conditioned (condition number: {cond_num})')
    u_f = np.linalg.solve(K_ff, F_f)
    u = np.zeros(n_dofs)
    r = np.zeros(n_dofs)
    u[free_dofs] = u_f
    r_c = K_cf @ u_f - F_c
    r[fixed_dofs] = r_c
    return (u, r)