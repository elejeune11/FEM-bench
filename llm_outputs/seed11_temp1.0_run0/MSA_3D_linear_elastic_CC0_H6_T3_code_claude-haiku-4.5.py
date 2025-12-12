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
    N = len(node_coords)
    ndof = 6 * N
    K_global = np.zeros((ndof, ndof))
    F_global = np.zeros(ndof)
    for (node_idx, loads) in nodal_loads.items():
        dof_start = node_idx * 6
        F_global[dof_start:dof_start + 6] = loads
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        coord_i = node_coords[node_i]
        coord_j = node_coords[node_j]
        L_vec = coord_j - coord_i
        L = np.linalg.norm(L_vec)
        if L < 1e-12:
            continue
        e_x = L_vec / L
        if 'local_z' in elem and elem['local_z'] is not None:
            e_z = np.array(elem['local_z'])
            e_z = e_z / np.linalg.norm(e_z)
        else:
            global_z = np.array([0.0, 0.0, 1.0])
            if abs(abs(np.dot(e_x, global_z)) - 1.0) < 1e-10:
                e_z = np.array([0.0, 1.0, 0.0])
            else:
                e_z = global_z
        e_z = e_z - np.dot(e_z, e_x) * e_x
        e_z = e_z / np.linalg.norm(e_z)
        e_y = np.cross(e_z, e_x)
        R = np.array([e_x, e_y, e_z])
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
        EI_y = E * I_y
        EI_z = E * I_z
        k_local[1, 1] = 12 * EI_z / L ** 3
        k_local[1, 5] = 6 * EI_z / L ** 2
        k_local[1, 7] = -12 * EI_z / L ** 3
        k_local[1, 11] = 6 * EI_z / L ** 2
        k_local[5, 1] = 6 * EI_z / L ** 2
        k_local[5, 5] = 4 * EI_z / L
        k_local[5, 7] = -6 * EI_z / L ** 2
        k_local[5, 11] = 2 * EI_z / L
        k_local[7, 1] = -12 * EI_z / L ** 3
        k_local[7, 5] = -6 * EI_z / L ** 2
        k_local[7, 7] = 12 * EI_z / L ** 3
        k_local[7, 11] = -6 * EI_z / L ** 2
        k_local[11, 1] = 6 * EI_z / L ** 2
        k_local[11, 5] = 2 * EI_z / L
        k_local[11, 7] = -6 * EI_z / L ** 2
        k_local[11, 11] = 4 * EI_z / L
        k_local[2, 2] = 12 * EI_y / L ** 3
        k_local[2, 4] = -6 * EI_y / L ** 2
        k_local[2, 8] = -12 * EI_y / L ** 3
        k_local[2, 10] = -6 * EI_y / L ** 2
        k_local[4, 2] = -6 * EI_y / L ** 2
        k_local[4, 4] = 4 * EI_y / L
        k_local[4, 8] = 6 * EI_y / L ** 2
        k_local[4, 10] = 2 * EI_y / L
        k_local[8, 2] = -12 * EI_y / L ** 3
        k_local[8, 4] = 6 * EI_y / L ** 2
        k_local[8, 8] = 12 * EI_y / L ** 3
        k_local[8, 10] = 6 * EI_y / L ** 2
        k_local[10, 2] = -6 * EI_y / L ** 2
        k_local[10, 4] = 2 * EI_y / L
        k_local[10, 8] = 6 * EI_y / L ** 2
        k_local[10, 10] = 4 * EI_y / L
        T = np.zeros((12, 12))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        k_global = T.T @ k_local @ T
        dof_i = np.arange(node_i * 6, node_i * 6 + 6)
        dof_j = np.arange(node_j * 6, node_j * 6 + 6)
        for (ii, dof_ii) in enumerate(dof_i):
            for (jj, dof_jj) in enumerate(dof_i):
                K_global[dof_ii, dof_jj] += k_global[ii, jj]
            for (jj, dof_jj) in enumerate(dof_j):
                K_global[dof_ii, dof_jj] += k_global[ii, jj + 6]
        for (ii, dof_ii) in enumerate(dof_j):
            for (jj, dof_jj) in enumerate(dof_i):
                K_global[dof_ii, dof_jj] += k_global[ii + 6, jj]
            for (jj, dof_jj) in enumerate(dof_j):
                K_global[dof_ii, dof_jj] += k_global[ii + 6, jj + 6]
    fixed_dofs = set()
    for (node_idx, bc) in boundary_conditions.items():
        for (dof_local, is_fixed) in enumerate(bc):
            if is_fixed:
                fixed_dofs.add(node_idx * 6 + dof_local)
    free_dofs = np.array(sorted(set(range(ndof)) - fixed_dofs))
    fixed_dofs = np.array(sorted(fixed_dofs))
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_fc = K_global[np.ix_(free_dofs, fixed_dofs)]
    K_cf = K_global[np.ix_(fixed_dofs, free_dofs)]
    K_cc = K_global[np.ix_(fixed_dofs, fixed_dofs)]
    F_f = F_global[free_dofs]
    try:
        cond = np.linalg.cond(K_ff)
        if cond >= 1e+16:
            raise ValueError(f'Ill-conditioned stiffness matrix (condition number = {cond})')
    except np.linalg.LinAlgError:
        raise ValueError('Singular stiffness matrix')
    u_f = np.linalg.solve(K_ff, F_f)
    u = np.zeros(ndof)
    u[free_dofs] = u_f
    r = np.zeros(ndof)
    r[fixed_dofs] = K_cf @ u_f - F_global[fixed_dofs]
    return (u, r)