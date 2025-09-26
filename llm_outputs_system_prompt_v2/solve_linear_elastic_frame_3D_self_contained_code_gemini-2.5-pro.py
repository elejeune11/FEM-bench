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
    num_nodes = len(node_coords)
    num_dofs = 6 * num_nodes
    K = np.zeros((num_dofs, num_dofs))
    P = np.zeros(num_dofs)
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        (E, nu, A) = (elem['E'], elem['nu'], elem['A'])
        (Iy, Iz, J) = (elem['I_y'], elem['I_z'], elem['J'])
        G = E / (2.0 * (1.0 + nu))
        p1 = node_coords[i]
        p2 = node_coords[j]
        vec = p2 - p1
        L = np.linalg.norm(vec)
        if L < 1e-12:
            continue
        k_local = np.zeros((12, 12))
        k_local[0, 0] = k_local[6, 6] = E * A / L
        k_local[0, 6] = k_local[6, 0] = -E * A / L
        k_local[3, 3] = k_local[9, 9] = G * J / L
        k_local[3, 9] = k_local[9, 3] = -G * J / L
        (c1, c2) = (12 * E * Iz / L ** 3, 6 * E * Iz / L ** 2)
        (c3, c4) = (4 * E * Iz / L, 2 * E * Iz / L)
        k_local[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])] = np.array([[c1, c2, -c1, c2], [c2, c3, -c2, c4], [-c1, -c2, c1, -c2], [c2, c4, -c2, c3]])
        (c5, c6) = (12 * E * Iy / L ** 3, 6 * E * Iy / L ** 2)
        (c7, c8) = (4 * E * Iy / L, 2 * E * Iy / L)
        k_local[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])] = np.array([[c5, -c6, -c5, -c6], [-c6, c7, c6, c8], [-c5, c6, c5, c6], [-c6, c8, c6, c7]])
        x_local = vec / L
        local_z_vec = elem.get('local_z')
        if local_z_vec is not None:
            v_ref = np.array(local_z_vec, dtype=float)
        else:
            v_ref = np.array([0.0, 0.0, 1.0])
            if np.abs(np.dot(x_local, v_ref)) > 1.0 - 1e-09:
                v_ref = np.array([0.0, 1.0, 0.0])
        y_dir = np.cross(x_local, v_ref)
        y_local = y_dir / np.linalg.norm(y_dir)
        z_local = np.cross(x_local, y_local)
        R = np.vstack([x_local, y_local, z_local])
        T_full = np.zeros((12, 12))
        for k in range(4):
            T_full[3 * k:3 * (k + 1), 3 * k:3 * (k + 1)] = R
        k_global = T_full.T @ k_local @ T_full
        dofs = np.concatenate((np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)))
        K[np.ix_(dofs, dofs)] += k_global
    for (node_idx, loads) in nodal_loads.items():
        P[6 * node_idx:6 * node_idx + 6] = loads
    fixed_dofs_mask = np.zeros(num_dofs, dtype=bool)
    for (node_idx, bcs) in boundary_conditions.items():
        start = 6 * node_idx
        fixed_dofs_mask[start:start + 6] = np.array(bcs, dtype=bool)
    free_dofs = np.where(~fixed_dofs_mask)[0]
    fixed_dofs = np.where(fixed_dofs_mask)[0]
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    u_f = np.zeros(len(free_dofs))
    if K_ff.shape[0] > 0:
        try:
            cond = np.linalg.cond(K_ff)
        except np.linalg.LinAlgError:
            cond = np.inf
        if cond >= 1e+16:
            raise ValueError(f'Free-free stiffness submatrix is ill-conditioned (condition number: {cond:.2e}).')
        u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(num_dofs)
    u[free_dofs] = u_f
    F_internal = K @ u
    r = np.zeros(num_dofs)
    r[fixed_dofs] = F_internal[fixed_dofs] - P[fixed_dofs]
    return (u, r)