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
    dof_per_node = 6
    total_dofs = num_nodes * dof_per_node
    K = np.zeros((total_dofs, total_dofs))
    P = np.zeros(total_dofs)
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
        T = np.eye(12)
        k_global = T.T @ k_local @ T
        dof_map = np.array([node_i * dof_per_node, node_i * dof_per_node + 1, node_i * dof_per_node + 2, node_i * dof_per_node + 3, node_i * dof_per_node + 4, node_i * dof_per_node + 5, node_j * dof_per_node, node_j * dof_per_node + 1, node_j * dof_per_node + 2, node_j * dof_per_node + 3, node_j * dof_per_node + 4, node_j * dof_per_node + 5])
        for i in range(12):
            for j in range(12):
                K[dof_map[i], dof_map[j]] += k_global[i, j]
    for (node, loads) in nodal_loads.items():
        for i in range(dof_per_node):
            P[node * dof_per_node + i] += loads[i]
    free_dofs = np.ones(total_dofs, dtype=bool)
    for (node, bc) in boundary_conditions.items():
        for i in range(dof_per_node):
            if bc[i] == 1:
                free_dofs[node * dof_per_node + i] = False
    fixed_dofs = ~free_dofs
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_fc = K[np.ix_(free_dofs, fixed_dofs)]
    P_f = P[free_dofs]
    if np.linalg.cond(K_ff) >= 1e+16:
        raise ValueError('The free-free stiffness submatrix K_ff is ill-conditioned.')
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(total_dofs)
    u[free_dofs] = u_f
    r = np.zeros(total_dofs)
    r[fixed_dofs] = K[fixed_dofs, :] @ u - P[fixed_dofs]
    return (u, r)