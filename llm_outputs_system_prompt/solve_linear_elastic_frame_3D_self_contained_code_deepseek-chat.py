def solve_linear_elastic_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    """
    Small-displacement linear-elastic analysis of a 3D frame using beam elements.
    The function assembles the global stiffness matrix (K) and load vector (P),
    partitions degrees of freedom (DOFs) into free and fixed sets, solves the
    reduced system for displacements at the free DOFs, and computes true support
    reactions at the fixed DOFs.
    Coordinate system: global right-handed Cartesian. Element local axes follow the
    beam axis (local x) with orientation defined via a reference vector.
    Condition number policy: the system is solved only if the free–free stiffness
    submatrix K_ff is well-conditioned (cond(K_ff) < 1e16). Otherwise a ValueError
    is raised.
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
            'local_z' : (3,) array or None
                Optional unit vector to define the local z-direction for transformation
                matrix orientation (must be unit length and not parallel to the beam axis).
    boundary_conditions : dict[int, Sequence[int]]
        Maps node index → 6-element iterable of 0 (free) or 1 (fixed). Omitted nodes ⇒ all DOFs free.
    nodal_loads : dict[int, Sequence[float]]
        Maps node index → 6-element [Fx, Fy, Fz, Mx, My, Mz]. Omitted nodes ⇒ zero loads.
    Returns
    -------
    u : (6*N,) ndarray
        Global displacement vector ordered as [UX, UY, UZ, RX, RY, RZ] for each node
        in sequence. Values are computed at free DOFs; fixed DOFs are zero.
    r : (6*N,) ndarray
        Global reaction force/moment vector with nonzeros only at fixed DOFs.
        Reactions are computed as internal elastic forces minus applied loads at the
        fixed DOFs; free DOFs have zero entries.
    Raises
    ------
    ValueError
        If the free-free submatrix K_ff is ill-conditioned (cond(K_ff) ≥ 1e16).
    Notes
    -----
    """
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K_global = np.zeros((n_dofs, n_dofs))
    P_global = np.zeros(n_dofs)
    for element in elements:
        i = element['node_i']
        j = element['node_j']
        E = element['E']
        nu = element['nu']
        A = element['A']
        I_y = element['I_y']
        I_z = element['I_z']
        J = element['J']
        local_z = element.get('local_z')
        coords_i = node_coords[i]
        coords_j = node_coords[j]
        L = np.linalg.norm(coords_j - coords_i)
        G = E / (2 * (1 + nu))
        k_local = np.zeros((12, 12))
        k_local[0, 0] = E * A / L
        k_local[0, 6] = -E * A / L
        k_local[6, 0] = -E * A / L
        k_local[6, 6] = E * A / L
        k_local[1, 1] = 12 * E * I_z / L ** 3
        k_local[1, 5] = 6 * E * I_z / L ** 2
        k_local[1, 7] = -12 * E * I_z / L ** 3
        k_local[1, 11] = 6 * E * I_z / L ** 2
        k_local[2, 2] = 12 * E * I_y / L ** 3
        k_local[2, 4] = -6 * E * I_y / L ** 2
        k_local[2, 8] = -12 * E * I_y / L ** 3
        k_local[2, 10] = -6 * E * I_y / L ** 2
        k_local[3, 3] = G * J / L
        k_local[3, 9] = -G * J / L
        k_local[9, 3] = -G * J / L
        k_local[9, 9] = G * J / L
        k_local[4, 2] = -6 * E * I_y / L ** 2
        k_local[4, 4] = 4 * E * I_y / L
        k_local[4, 8] = 6 * E * I_y / L ** 2
        k_local[4, 10] = 2 * E * I_y / L
        k_local[5, 1] = 6 * E * I_z / L ** 2
        k_local[5, 5] = 4 * E * I_z / L
        k_local[5, 7] = -6 * E * I_z / L ** 2
        k_local[5, 11] = 2 * E * I_z / L
        k_local[7, 1] = -12 * E * I_z / L ** 3
        k_local[7, 5] = -6 * E * I_z / L ** 2
        k_local[7, 7] = 12 * E * I_z / L ** 3
        k_local[7, 11] = -6 * E * I_z / L ** 2
        k_local[8, 2] = -12 * E * I_y / L ** 3
        k_local[8, 4] = 6 * E * I_y / L ** 2
        k_local[8, 8] = 12 * E * I_y / L ** 3
        k_local[8, 10] = 6 * E * I_y / L ** 2
        k_local[10, 2] = -6 * E * I_y / L ** 2
        k_local[10, 4] = 2 * E * I_y / L
        k_local[10, 8] = 6 * E * I_y / L ** 2
        k_local[10, 10] = 4 * E * I_y / L
        k_local[11, 1] = 6 * E * I_z / L ** 2
        k_local[11, 5] = 2 * E * I_z / L
        k_local[11, 7] = -6 * E * I_z / L ** 2
        k_local[11, 11] = 4 * E * I_z / L
        vec_x = (coords_j - coords_i) / L
        if local_z is not None:
            vec_z = np.array(local_z)
            vec_z = vec_z / np.linalg.norm(vec_z)
            vec_y = np.cross(vec_z, vec_x)
            vec_y = vec_y / np.linalg.norm(vec_y)
            vec_z = np.cross(vec_x, vec_y)
            vec_z = vec_z / np.linalg.norm(vec_z)
        else:
            if abs(vec_x[2]) > 0.9:
                vec_y = np.array([0.0, 1.0, 0.0])
            else:
                vec_y = np.array([0.0, 0.0, 1.0])
            vec_z = np.cross(vec_x, vec_y)
            vec_z = vec_z / np.linalg.norm(vec_z)
            vec_y = np.cross(vec_z, vec_x)
            vec_y = vec_y / np.linalg.norm(vec_y)
        T = np.zeros((3, 3))
        T[0, :] = vec_x
        T[1, :] = vec_y
        T[2, :] = vec_z
        R = np.zeros((12, 12))
        R[0:3, 0:3] = T
        R[3:6, 3:6] = T
        R[6:9, 6:9] = T
        R[9:12, 9:12] = T
        k_global_element = R.T @ k_local @ R
        dof_indices = []
        for node in [i, j]:
            for dof in range(6):
                dof_indices.append(6 * node + dof)
        for (idx_i, global_i) in enumerate(dof_indices):
            for (idx_j, global_j) in enumerate(dof_indices):
                K_global[global_i, global_j] += k_global_element[idx_i, idx_j]
    for (node, loads) in nodal_loads.items():
        for dof in range(6):
            P_global[6 * node + dof] += loads[dof]
    fixed_dofs = []
    free_dofs = []
    for node in range(n_nodes):
        bc = boundary_conditions.get(node, [0, 0, 0, 0, 0, 0])
        for dof in range(6):
            if bc[dof] == 1:
                fixed_dofs.append(6 * node + dof)
            else:
                free_dofs.append(6 * node + dof)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_fc = K_global[np.ix_(free_dofs, fixed_dofs)]
    P_f = P_global[free_dofs]
    cond_num = np.linalg.cond(K_ff)
    if cond_num >= 1e+16:
        raise ValueError('Free-free submatrix is ill-conditioned')
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(n_dofs)
    u[free_dofs] = u_f
    if fixed_dofs:
        P_c = P_global[fixed_dofs]
        u_c = u[fixed_dofs]
        r_c = K_global[np.ix_(fixed_dofs, free_dofs)] @ u_f + K_global[np.ix_(fixed_dofs, fixed_dofs)] @ u_c - P_c
    else:
        r_c = np.array([])
    r = np.zeros(n_dofs)
    if fixed_dofs:
        r[fixed_dofs] = r_c
    return (u, r)