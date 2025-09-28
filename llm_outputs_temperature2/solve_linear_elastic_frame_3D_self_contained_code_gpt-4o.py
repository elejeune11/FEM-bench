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
    num_nodes = node_coords.shape[0]
    num_dofs = 6 * num_nodes
    K = np.zeros((num_dofs, num_dofs))
    P = np.zeros(num_dofs)
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        E = element['E']
        nu = element['nu']
        A = element['A']
        I_y = element['I_y']
        I_z = element['I_z']
        J = element['J']
        local_z = element['local_z']
        L = np.linalg.norm(node_coords[node_j] - node_coords[node_i])
        G = E / (2 * (1 + nu))
        k_local = np.zeros((12, 12))
        k_local[0, 0] = k_local[6, 6] = A * E / L
        k_local[0, 6] = k_local[6, 0] = -A * E / L
        k_local[1, 1] = k_local[7, 7] = 12 * E * I_z / L ** 3
        k_local[1, 7] = k_local[7, 1] = -12 * E * I_z / L ** 3
        k_local[2, 2] = k_local[8, 8] = 12 * E * I_y / L ** 3
        k_local[2, 8] = k_local[8, 2] = -12 * E * I_y / L ** 3
        k_local[3, 3] = k_local[9, 9] = G * J / L
        k_local[3, 9] = k_local[9, 3] = -G * J / L
        k_local[4, 4] = k_local[10, 10] = 4 * E * I_y / L
        k_local[4, 10] = k_local[10, 4] = 2 * E * I_y / L
        k_local[5, 5] = k_local[11, 11] = 4 * E * I_z / L
        k_local[5, 11] = k_local[11, 5] = 2 * E * I_z / L
        k_local[1, 5] = k_local[5, 1] = 6 * E * I_z / L ** 2
        k_local[1, 11] = k_local[11, 1] = -6 * E * I_z / L ** 2
        k_local[2, 4] = k_local[4, 2] = -6 * E * I_y / L ** 2
        k_local[2, 10] = k_local[10, 2] = 6 * E * I_y / L ** 2
        k_local[4, 8] = k_local[8, 4] = -6 * E * I_y / L ** 2
        k_local[5, 7] = k_local[7, 5] = 6 * E * I_z / L ** 2
        x_local = (node_coords[node_j] - node_coords[node_i]) / L
        if local_z is None:
            local_z = np.array([0, 0, 1])
        y_local = np.cross(local_z, x_local)
        y_local /= np.linalg.norm(y_local)
        z_local = np.cross(x_local, y_local)
        T = np.zeros((12, 12))
        T[:3, :3] = T[3:6, 3:6] = T[6:9, 6:9] = T[9:, 9:] = np.vstack([x_local, y_local, z_local]).T
        k_global = T.T @ k_local @ T
        dof_map = np.array([6 * node_i, 6 * node_i + 1, 6 * node_i + 2, 6 * node_i + 3, 6 * node_i + 4, 6 * node_i + 5, 6 * node_j, 6 * node_j + 1, 6 * node_j + 2, 6 * node_j + 3, 6 * node_j + 4, 6 * node_j + 5])
        for i in range(12):
            for j in range(12):
                K[dof_map[i], dof_map[j]] += k_global[i, j]
    for (node, loads) in nodal_loads.items():
        P[6 * node:6 * node + 6] = loads
    free_dofs = np.ones(num_dofs, dtype=bool)
    for (node, bc) in boundary_conditions.items():
        for (i, fixed) in enumerate(bc):
            if fixed:
                free_dofs[6 * node + i] = False
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_fc = K[np.ix_(free_dofs, ~free_dofs)]
    P_f = P[free_dofs]
    P_c = P[~free_dofs]
    if np.linalg.cond(K_ff) >= 1e+16:
        raise ValueError('The free-free submatrix K_ff is ill-conditioned.')
    u_f = np.linalg.solve(K_ff, P_f)
    r_c = K_fc.T @ u_f - P_c
    u = np.zeros(num_dofs)
    r = np.zeros(num_dofs)
    u[free_dofs] = u_f
    r[~free_dofs] = r_c
    return (u, r)