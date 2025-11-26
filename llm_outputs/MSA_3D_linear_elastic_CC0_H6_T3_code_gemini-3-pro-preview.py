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
    n_dof = 6 * n_nodes
    K_global = np.zeros((n_dof, n_dof))
    for el in elements:
        i_idx = el['node_i']
        j_idx = el['node_j']
        p_i = node_coords[i_idx]
        p_j = node_coords[j_idx]
        delta = p_j - p_i
        L = np.linalg.norm(delta)
        if L < 1e-14:
            raise ValueError('Element has zero or near-zero length.')
        x_local = delta / L
        if 'local_z' in el and el['local_z'] is not None:
            v_ref = np.array(el['local_z'], dtype=float)
            v_ref = v_ref / np.linalg.norm(v_ref)
        else:
            global_z = np.array([0.0, 0.0, 1.0])
            if np.linalg.norm(np.cross(x_local, global_z)) < 1e-10:
                v_ref = np.array([0.0, 1.0, 0.0])
            else:
                v_ref = global_z
        cross_vx = np.cross(v_ref, x_local)
        norm_cross = np.linalg.norm(cross_vx)
        if norm_cross < 1e-14:
            raise ValueError('local_z reference vector is parallel to the beam axis.')
        y_local = cross_vx / norm_cross
        z_local = np.cross(x_local, y_local)
        r_3x3 = np.stack([x_local, y_local, z_local])
        R = np.zeros((12, 12))
        for blk in range(4):
            R[3 * blk:3 * blk + 3, 3 * blk:3 * blk + 3] = r_3x3
        E = el['E']
        nu = el['nu']
        A = el['A']
        Iy = el['I_y']
        Iz = el['I_z']
        J = el['J']
        G = E / (2.0 * (1.0 + nu))
        k_local = np.zeros((12, 12))
        k_a = E * A / L
        k_local[0, 0] = k_a
        k_local[0, 6] = -k_a
        k_local[6, 0] = -k_a
        k_local[6, 6] = k_a
        k_t = G * J / L
        k_local[3, 3] = k_t
        k_local[3, 9] = -k_t
        k_local[9, 3] = -k_t
        k_local[9, 9] = k_t
        k_bz1 = 12 * E * Iz / L ** 3
        k_bz2 = 6 * E * Iz / L ** 2
        k_bz3 = 4 * E * Iz / L
        k_bz4 = 2 * E * Iz / L
        idx_z = [1, 5, 7, 11]
        sub_z = np.array([[k_bz1, k_bz2, -k_bz1, k_bz2], [k_bz2, k_bz3, -k_bz2, k_bz4], [-k_bz1, -k_bz2, k_bz1, -k_bz2], [k_bz2, k_bz4, -k_bz2, k_bz3]])
        for (r_i, r_glob) in enumerate(idx_z):
            for (c_i, c_glob) in enumerate(idx_z):
                k_local[r_glob, c_glob] += sub_z[r_i, c_i]
        k_by1 = 12 * E * Iy / L ** 3
        k_by2 = 6 * E * Iy / L ** 2
        k_by3 = 4 * E * Iy / L
        k_by4 = 2 * E * Iy / L
        idx_y = [2, 4, 8, 10]
        sub_y = np.array([[k_by1, -k_by2, -k_by1, -k_by2], [-k_by2, k_by3, k_by2, k_by4], [-k_by1, k_by2, k_by1, k_by2], [-k_by2, k_by4, k_by2, k_by3]])
        for (r_i, r_glob) in enumerate(idx_y):
            for (c_i, c_glob) in enumerate(idx_y):
                k_local[r_glob, c_glob] += sub_y[r_i, c_i]
        k_global_el = R.T @ k_local @ R
        global_indices = []
        for n in [i_idx, j_idx]:
            start = n * 6
            global_indices.extend(range(start, start + 6))
        for (r_i, r_g) in enumerate(global_indices):
            for (c_i, c_g) in enumerate(global_indices):
                K_global[r_g, c_g] += k_global_el[r_i, c_i]
    F_global = np.zeros(n_dof)
    for (node_idx, loads) in nodal_loads.items():
        start = node_idx * 6
        F_global[start:start + 6] = loads
    fixed_dofs_list = []
    for (node_idx, conditions) in boundary_conditions.items():
        start = node_idx * 6
        for (i, is_fixed) in enumerate(conditions):
            if is_fixed:
                fixed_dofs_list.append(start + i)
    fixed_dofs = np.array(sorted(fixed_dofs_list), dtype=int)
    all_dofs = np.arange(n_dof)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
    u_global = np.zeros(n_dof)
    r_global = np.zeros(n_dof)
    if len(free_dofs) > 0:
        K_ff = K_global[np.ix_(free_dofs, free_dofs)]
        F_f = F_global[free_dofs]
        cond_num = np.linalg.cond(K_ff)
        if cond_num >= 1e+16:
            raise ValueError(f'Stiffness matrix is ill-conditioned (condition number {cond_num})')
        u_free = np.linalg.solve(K_ff, F_f)
        u_global[free_dofs] = u_free
    F_internal = K_global @ u_global
    r_total = F_internal - F_global
    if len(free_dofs) > 0:
        r_total[free_dofs] = 0.0
    return (u_global, r_total)