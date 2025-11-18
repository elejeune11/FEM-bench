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
        vec_z_ref = elem.get('local_z')
        p1 = node_coords[i]
        p2 = node_coords[j]
        vec_ij = p2 - p1
        L = np.linalg.norm(vec_ij)
        if L < 1e-09:
            continue
        lx = vec_ij / L
        if vec_z_ref is None:
            if abs(lx[2]) > 0.9999:
                v_ref = np.array([0.0, 1.0, 0.0])
            else:
                v_ref = np.array([0.0, 0.0, 1.0])
        else:
            v_ref = np.array(vec_z_ref)
        ly = np.cross(v_ref, lx)
        if np.linalg.norm(ly) < 1e-09:
            v_ref = np.array([1.0, 0.0, 0.0])
            ly = np.cross(v_ref, lx)
        ly /= np.linalg.norm(ly)
        lz = np.cross(lx, ly)
        R = np.vstack([lx, ly, lz])
        T_block = np.zeros((12, 12))
        for k in range(4):
            T_block[3 * k:3 * (k + 1), 3 * k:3 * (k + 1)] = R
        G = E / (2 * (1 + nu))
        k_local = np.zeros((12, 12))
        k_local[0, 0] = k_local[6, 6] = E * A / L
        k_local[0, 6] = k_local[6, 0] = -E * A / L
        k_local[3, 3] = k_local[9, 9] = G * J / L
        k_local[3, 9] = k_local[9, 3] = -G * J / L
        c1z = 12 * E * Iz / L ** 3
        c2z = 6 * E * Iz / L ** 2
        c3z = 4 * E * Iz / L
        c4z = 2 * E * Iz / L
        k_bending_xy = np.array([[c1z, c2z, -c1z, c2z], [c2z, c3z, -c2z, c4z], [-c1z, -c2z, c1z, -c2z], [c2z, c4z, -c2z, c3z]])
        dof_map_xy = [1, 5, 7, 11]
        k_local[np.ix_(dof_map_xy, dof_map_xy)] = k_bending_xy
        c1y = 12 * E * Iy / L ** 3
        c2y = 6 * E * Iy / L ** 2
        c3y = 4 * E * Iy / L
        c4y = 2 * E * Iy / L
        k_bending_xz = np.array([[c1y, -c2y, -c1y, -c2y], [-c2y, c3y, c2y, c4y], [-c1y, c2y, c1y, c2y], [-c2y, c4y, c2y, c3y]])
        dof_map_xz = [2, 4, 8, 10]
        k_local[np.ix_(dof_map_xz, dof_map_xz)] = k_bending_xz
        k_global = T_block.T @ k_local @ T_block
        dof_indices = np.concatenate([np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)])
        K[np.ix_(dof_indices, dof_indices)] += k_global
    for (node_idx, loads) in nodal_loads.items():
        start_dof = 6 * node_idx
        P[start_dof:start_dof + 6] += loads
    fixed_dofs = []
    for (node_idx, bcs) in boundary_conditions.items():
        for i in range(6):
            if bcs[i] == 1:
                fixed_dofs.append(6 * node_idx + i)
    all_dofs = list(range(num_dofs))
    free_dofs = sorted(list(set(all_dofs) - set(fixed_dofs)))
    fixed_dofs = sorted(fixed_dofs)
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    u_f = np.array([])
    if K_ff.shape[0] > 0:
        cond = np.linalg.cond(K_ff)
        if cond >= 1e+16:
            raise ValueError('The free-free submatrix K_ff is ill-conditioned (cond(K_ff) >= 1e16).')
        u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(num_dofs)
    if len(free_dofs) > 0:
        u[free_dofs] = u_f
    r = np.zeros(num_dofs)
    if len(fixed_dofs) > 0:
        K_sf = K[np.ix_(fixed_dofs, free_dofs)]
        P_s = P[fixed_dofs]
        r_s = -P_s
        if len(free_dofs) > 0:
            r_s += K_sf @ u_f
        r[fixed_dofs] = r_s
    return (u, r)