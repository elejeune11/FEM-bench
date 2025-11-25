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
    num_dofs = 6 * N
    K = np.zeros((num_dofs, num_dofs))
    P = np.zeros(num_dofs)
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        (E, nu, A) = (elem['E'], elem['nu'], elem['A'])
        (Iy, Iz, J) = (elem['I_y'], elem['I_z'], elem['J'])
        G = E / (2 * (1 + nu))
        p_i = node_coords[i]
        p_j = node_coords[j]
        vec_ij = p_j - p_i
        L = np.linalg.norm(vec_ij)
        if np.isclose(L, 0):
            continue
        local_x = vec_ij / L
        ref_vec_in = elem.get('local_z')
        if ref_vec_in is None:
            ref_vec = np.array([0.0, 0.0, 1.0])
            if np.allclose(np.abs(np.dot(local_x, ref_vec)), 1.0):
                ref_vec = np.array([0.0, 1.0, 0.0])
        else:
            ref_vec = np.asarray(ref_vec_in)
        temp_y = np.cross(local_x, ref_vec)
        local_y = temp_y / np.linalg.norm(temp_y)
        local_z = np.cross(local_x, local_y)
        R = np.column_stack((local_x, local_y, local_z))
        T = np.kron(np.eye(4), R)
        k_local = np.zeros((12, 12))
        k_local[0, 0] = k_local[6, 6] = E * A / L
        k_local[0, 6] = k_local[6, 0] = -E * A / L
        k_local[3, 3] = k_local[9, 9] = G * J / L
        k_local[3, 9] = k_local[9, 3] = -G * J / L
        (c1z, c2z, c3z, c4z) = (12 * E * Iz / L ** 3, 6 * E * Iz / L ** 2, 4 * E * Iz / L, 2 * E * Iz / L)
        k_local[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])] = np.array([[c1z, c2z, -c1z, c2z], [c2z, c3z, -c2z, c4z], [-c1z, -c2z, c1z, -c2z], [c2z, c4z, -c2z, c3z]])
        (c1y, c2y, c3y, c4y) = (12 * E * Iy / L ** 3, 6 * E * Iy / L ** 2, 4 * E * Iy / L, 2 * E * Iy / L)
        k_local[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])] = np.array([[c1y, -c2y, -c1y, -c2y], [-c2y, c3y, c2y, c4y], [-c1y, c2y, c1y, c2y], [-c2y, c4y, c2y, c3y]])
        k_global = T @ k_local @ T.T
        dofs = np.concatenate((np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)))
        K[np.ix_(dofs, dofs)] += k_global
    for (node_idx, loads) in nodal_loads.items():
        start_dof = 6 * node_idx
        P[start_dof:start_dof + 6] += loads
    fixed_dofs_list = []
    for (node_idx, bc_flags) in boundary_conditions.items():
        for (dof_idx, flag) in enumerate(bc_flags):
            if flag == 1:
                fixed_dofs_list.append(6 * node_idx + dof_idx)
    all_dofs = np.arange(num_dofs)
    fixed_dofs = np.unique(fixed_dofs_list)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs, assume_unique=True)
    if free_dofs.size == 0:
        u = np.zeros(num_dofs)
        r = np.zeros(num_dofs)
        if fixed_dofs.size > 0:
            r[fixed_dofs] = -P[fixed_dofs]
        return (u, r)
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    if np.linalg.cond(K_ff) >= 1e+16:
        raise ValueError('The free-free submatrix K_ff is ill-conditioned (cond(K_ff) >= 1e16).')
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(num_dofs)
    u[free_dofs] = u_f
    r = np.zeros(num_dofs)
    if fixed_dofs.size > 0:
        K_sf = K[np.ix_(fixed_dofs, free_dofs)]
        P_s = P[fixed_dofs]
        r_s = K_sf @ u_f - P_s
        r[fixed_dofs] = r_s
    return (u, r)