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
    for el in elements:
        (i, j) = (el['node_i'], el['node_j'])
        (E, nu, A) = (el['E'], el['nu'], el['A'])
        (Iy, Iz, J) = (el['I_y'], el['I_z'], el['J'])
        G = E / (2 * (1 + nu))
        p1 = node_coords[i]
        p2 = node_coords[j]
        vec = p2 - p1
        L = np.linalg.norm(vec)
        if L < 1e-09:
            continue
        k_local = np.zeros((12, 12))
        EAL = E * A / L
        GJL = G * J / L
        EIyL = E * Iy / L
        EIyL2 = E * Iy / L ** 2
        EIyL3 = E * Iy / L ** 3
        EIzL = E * Iz / L
        EIzL2 = E * Iz / L ** 2
        EIzL3 = E * Iz / L ** 3
        k_local[0, 0] = k_local[6, 6] = EAL
        k_local[0, 6] = k_local[6, 0] = -EAL
        k_local[3, 3] = k_local[9, 9] = GJL
        k_local[3, 9] = k_local[9, 3] = -GJL
        k_local[1, 1] = k_local[7, 7] = 12 * EIzL3
        k_local[1, 7] = k_local[7, 1] = -12 * EIzL3
        k_local[1, 5] = k_local[5, 1] = k_local[1, 11] = k_local[11, 1] = 6 * EIzL2
        k_local[5, 7] = k_local[7, 5] = k_local[11, 7] = k_local[7, 11] = -6 * EIzL2
        k_local[5, 5] = k_local[11, 11] = 4 * EIzL
        k_local[5, 11] = k_local[11, 5] = 2 * EIzL
        k_local[2, 2] = k_local[8, 8] = 12 * EIyL3
        k_local[2, 8] = k_local[8, 2] = -12 * EIyL3
        k_local[2, 4] = k_local[4, 2] = -6 * EIyL2
        k_local[2, 10] = k_local[10, 2] = -6 * EIyL2
        k_local[4, 8] = k_local[8, 4] = 6 * EIyL2
        k_local[8, 10] = k_local[10, 8] = 6 * EIyL2
        k_local[4, 4] = k_local[10, 10] = 4 * EIyL
        k_local[4, 10] = k_local[10, 4] = 2 * EIyL
        x_vec = vec / L
        v_ref = el.get('local_z')
        is_parallel = False
        if v_ref is not None:
            v_ref = np.asarray(v_ref, dtype=float)
            if np.linalg.norm(np.cross(x_vec, v_ref)) < 1e-08:
                is_parallel = True
        if v_ref is None or is_parallel:
            if abs(x_vec[0]) < 1e-08 and abs(x_vec[1]) < 1e-08:
                v_ref = np.array([0.0, 1.0, 0.0])
            else:
                v_ref = np.array([0.0, 0.0, 1.0])
        z_proj = v_ref - np.dot(v_ref, x_vec) * x_vec
        z_vec = z_proj / np.linalg.norm(z_proj)
        y_vec = np.cross(z_vec, x_vec)
        R = np.vstack((x_vec, y_vec, z_vec)).T
        T = np.zeros((12, 12))
        for k in range(4):
            T[3 * k:3 * (k + 1), 3 * k:3 * (k + 1)] = R
        k_global = T @ k_local @ T.T
        dof_indices = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
        ix = np.ix_(dof_indices, dof_indices)
        K[ix] += k_global
    for (node_idx, loads) in nodal_loads.items():
        start_dof = 6 * node_idx
        P[start_dof:start_dof + 6] += loads
    all_dofs = np.arange(num_dofs)
    is_fixed = np.zeros(num_dofs, dtype=bool)
    for (node_idx, bcs) in boundary_conditions.items():
        for dof_idx in range(6):
            if bcs[dof_idx] == 1:
                dof = 6 * node_idx + dof_idx
                is_fixed[dof] = True
    free_dofs = all_dofs[~is_fixed]
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    if K_ff.shape[0] > 0:
        cond_K_ff = np.linalg.cond(K_ff)
        if np.isinf(cond_K_ff) or cond_K_ff >= 1e+16:
            raise ValueError('The free-free submatrix K_ff is ill-conditioned (cond(K_ff) >= 1e16).')
        u_f = np.linalg.solve(K_ff, P_f)
    else:
        u_