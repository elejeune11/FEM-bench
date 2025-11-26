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
    num_nodes = len(node_coords)
    total_dofs = 6 * num_nodes
    K = np.zeros((total_dofs, total_dofs))
    F = np.zeros(total_dofs)
    for (node_idx, loads) in nodal_loads.items():
        start_dof = 6 * node_idx
        F[start_dof:start_dof + 6] = loads
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        (E, nu, A) = (elem['E'], elem['nu'], elem['A'])
        (I_y, I_z, J) = (elem['I_y'], elem['I_z'], elem['J'])
        G = E / (2 * (1 + nu))
        p1 = node_coords[i]
        p2 = node_coords[j]
        vec_x = p2 - p1
        L = np.linalg.norm(vec_x)
        if L < 1e-09:
            continue
        x_local = vec_x / L
        local_z_vec = elem.get('local_z')
        if local_z_vec is None:
            z_global = np.array([0.0, 0.0, 1.0])
            if np.abs(np.dot(x_local, z_global)) > 1.0 - 1e-09:
                ref_vec = np.array([0.0, 1.0, 0.0])
            else:
                ref_vec = z_global
        else:
            ref_vec = np.asarray(local_z_vec)
        y_local = np.cross(ref_vec, x_local)
        y_local /= np.linalg.norm(y_local)
        z_local = np.cross(x_local, y_local)
        z_local /= np.linalg.norm(z_local)
        R = np.vstack([x_local, y_local, z_local])
        T = np.zeros((12, 12))
        for k in range(4):
            T[3 * k:3 * (k + 1), 3 * k:3 * (k + 1)] = R
        k_local = np.zeros((12, 12))
        k_axial = E * A / L
        k_local[np.ix_([0, 6], [0, 6])] = k_axial * np.array([[1, -1], [-1, 1]])
        k_torsion = G * J / L
        k_local[np.ix_([3, 9], [3, 9])] = k_torsion * np.array([[1, -1], [-1, 1]])
        c1z = 12 * E * I_z / L ** 3
        c2z = 6 * E * I_z / L ** 2
        c3z = 4 * E * I_z / L
        c4z = 2 * E * I_z / L
        k_local[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])] = [[c1z, c2z, -c1z, c2z], [c2z, c3z, -c2z, c4z], [-c1z, -c2z, c1z, -c2z], [c2z, c4z, -c2z, c3z]]
        c1y = 12 * E * I_y / L ** 3
        c2y = 6 * E * I_y / L ** 2
        c3y = 4 * E * I_y / L
        c4y = 2 * E * I_y / L
        k_local[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])] = [[c1y, -c2y, -c1y, -c2y], [-c2y, c3y, c2y, c4y], [-c1y, c2y, c1y, c2y], [-c2y, c4y, c2y, c3y]]
        k_global = T.T @ k_local @ T
        global_indices = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
        K[np.ix_(global_indices, global_indices)] += k_global
    fixed_dofs_mask = np.zeros(total_dofs, dtype=bool)
    for (node_idx, bc_flags) in boundary_conditions.items():
        start_dof = 6 * node_idx
        fixed_dofs_mask[start_dof:start_dof + 6] = [bool(f) for f in bc_flags]
    free_dofs = np.where(~fixed_dofs_mask)[0]
    fixed_dofs = np.where(fixed_dofs_mask)[0]
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]
    u = np.zeros(total_dofs)
    if K_ff.shape[0] > 0:
        if np.linalg.cond(K_ff) > 1e+16:
            raise ValueError('The free-free stiffness matrix is ill-conditioned and the system cannot be reliably solved.')
        u_f = np.linalg.solve(K_ff, F_f)
        u[free_dofs] = u_f
    else:
        u_f = np.array([])
    r = np.zeros(total_dofs)
    if fixed_dofs.size > 0:
        F_s = F[fixed_dofs]
        if free_dofs.size > 0:
            K_sf = K[np.ix_(fixed_dofs, free_dofs)]
            r_s = K_sf @ u_f - F_s
        else:
            r_s = -F_s
        r[fixed_dofs] = r_s
    return (u, r)