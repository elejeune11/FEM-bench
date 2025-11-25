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
    N = node_coords.shape[0]
    total_dofs = 6 * N
    K = np.zeros((total_dofs, total_dofs))
    P = np.zeros(total_dofs)

    def get_dofs(node_idx):
        return np.arange(6 * node_idx, 6 * node_idx + 6)
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        Iy = elem['I_y']
        Iz = elem['I_z']
        J = elem['J']
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
        T = np.eye(12)
        k_global = T.T @ k_local @ T
        dofs_i = get_dofs(node_i)
        dofs_j = get_dofs(node_j)
        dofs = np.concatenate([dofs_i, dofs_j])
        for (i, dof_i) in enumerate(dofs):
            for (j, dof_j) in enumerate(dofs):
                K[dof_i, dof_j] += k_global[i, j]
    for (node_idx, loads) in nodal_loads.items():
        dofs = get_dofs(node_idx)
        P[dofs] = loads
    fixed_dofs = []
    for (node_idx, bc) in boundary_conditions.items():
        node_dofs = get_dofs(node_idx)
        for (i, is_fixed) in enumerate(bc):
            if is_fixed:
                fixed_dofs.append(node_dofs[i])
    fixed_dofs = np.array(fixed_dofs, dtype=int)
    all_dofs = np.arange(total_dofs)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    if np.linalg.cond(K_ff) >= 1e+16:
        raise ValueError('Free-free submatrix K_ff is ill-conditioned')
    u = np.zeros(total_dofs)
    if len(free_dofs) > 0:
        u[free_dofs] = np.linalg.solve(K_ff, P[free_dofs])
    r = np.zeros(total_dofs)
    if len(fixed_dofs) > 0:
        r[fixed_dofs] = K[np.ix_(fixed_dofs, all_dofs)] @ u - P[fixed_dofs]
    return (u, r)