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
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        (E, nu, A) = (elem['E'], elem['nu'], elem['A'])
        (Iy, Iz, J) = (elem['I_y'], elem['I_z'], elem['J'])
        G = E / (2 * (1 + nu))
        p1 = node_coords[i]
        p2 = node_coords[j]
        vec = p2 - p1
        L = np.linalg.norm(vec)
        if L == 0:
            continue
        x_prime = vec / L
        ref_vec_input = elem.get('local_z')
        if ref_vec_input is not None:
            v = np.array(ref_vec_input, dtype=float)
        else:
            v = np.array([0.0, 0.0, 1.0])
            if np.allclose(np.abs(np.dot(x_prime, v)), 1.0):
                v = np.array([0.0, 1.0, 0.0])
        y_prime = np.cross(v, x_prime)
        y_prime /= np.linalg.norm(y_prime)
        z_prime = np.cross(x_prime, y_prime)
        R = np.vstack([x_prime, y_prime, z_prime])
        T = np.zeros((12, 12))
        for k in range(4):
            T[3 * k:3 * (k + 1), 3 * k:3 * (k + 1)] = R
        k1 = E * A / L
        k2 = 12 * E * Iz / L ** 3
        k3 = 6 * E * Iz / L ** 2
        k4 = 4 * E * Iz / L
        k5 = 2 * E * Iz / L
        k6 = 12 * E * Iy / L ** 3
        k7 = 6 * E * Iy / L ** 2
        k8 = 4 * E * Iy / L
        k9 = 2 * E * Iy / L
        k10 = G * J / L
        k_local = np.array([[k1, 0, 0, 0, 0, 0, -k1, 0, 0, 0, 0, 0], [0, k2, 0, 0, 0, k3, 0, -k2, 0, 0, 0, k3], [0, 0, k6, 0, -k7, 0, 0, 0, -k6, 0, -k7, 0], [0, 0, 0, k10, 0, 0, 0, 0, 0, -k10, 0, 0], [0, 0, -k7, 0, k8, 0, 0, 0, k7, 0, k9, 0], [0, k3, 0, 0, 0, k4, 0, -k3, 0, 0, 0, k5], [-k1, 0, 0, 0, 0, 0, k1, 0, 0, 0, 0, 0], [0, -k2, 0, 0, 0, -k3, 0, k2, 0, 0, 0, -k3], [0, 0, -k6, 0, k7, 0, 0, 0, k6, 0, k7, 0], [0, 0, 0, -k10, 0, 0, 0, 0, 0, k10, 0, 0], [0, 0, -k7, 0, k9, 0, 0, 0, k7, 0, k8, 0], [0, k3, 0, 0, 0, k5, 0, -k3, 0, 0, 0, k4]])
        k_global = T.T @ k_local @ T
        dof_indices = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
        ix = np.ix_(dof_indices, dof_indices)
        K[ix] += k_global
    for (node_idx, loads) in nodal_loads.items():
        start_dof = 6 * node_idx
        P[start_dof:start_dof + 6] += loads
    all_dofs = np.arange(num_dofs)
    fixed_mask = np.zeros(num_dofs, dtype=bool)
    for (node_idx, bc_flags) in boundary_conditions.items():
        for (dof_offset, flag) in enumerate(bc_flags):
            if flag == 1:
                fixed_mask[6 * node_idx + dof_offset] = True
    free_dofs = all_dofs[~fixed_mask]
    fixed_dofs = all_dofs[fixed_mask]
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    u_f = np.array([])
    if K_ff.shape[0] > 0:
        try:
            cond_K_ff = np.linalg.cond(K_ff)
        except np.linalg.LinAlgError:
            cond_K_ff = np.inf
        if cond_K_ff >= 1e+16:
            raise ValueError('The free-free submatrix K_ff is ill-conditioned (cond(K_ff) >= 1e16).')
        u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(num_dofs)
    if u_f.size > 0:
        u[free_dofs] = u_f
    R_full = K @ u - P
    r = np.zeros(num_dofs)
    if fixed_dofs.size > 0:
        r[fixed_dofs] = R_full[fixed_dofs]
    return (u, r)