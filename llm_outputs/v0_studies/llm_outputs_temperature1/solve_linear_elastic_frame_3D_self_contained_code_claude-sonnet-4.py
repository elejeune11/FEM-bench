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
    N = len(node_coords)
    ndof = 6 * N
    K = np.zeros((ndof, ndof))
    P = np.zeros(ndof)
    for (node_idx, loads) in nodal_loads.items():
        for (i, load) in enumerate(loads):
            P[6 * node_idx + i] = load
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        E = element['E']
        nu = element['nu']
        A = element['A']
        I_y = element['I_y']
        I_z = element['I_z']
        J = element['J']
        local_z = element.get('local_z', None)
        (xi, yi, zi) = node_coords[node_i]
        (xj, yj, zj) = node_coords[node_j]
        (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        ex = np.array([dx, dy, dz]) / L
        if local_z is not None:
            ez = np.array(local_z)
            ez = ez / np.linalg.norm(ez)
        elif abs(ex[2]) < 0.9:
            ez = np.array([0, 0, 1])
        else:
            ez = np.array([1, 0, 0])
        ez = ez - np.dot(ez, ex) * ex
        ez = ez / np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        T = np.zeros((12, 12))
        R = np.array([ex, ey, ez]).T
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        G = E / (2 * (1 + nu))
        k_local = np.zeros((12, 12))
        k_local[0, 0] = E * A / L
        k_local[0, 6] = -E * A / L
        k_local[6, 0] = -E * A / L
        k_local[6, 6] = E * A / L
        k_local[2, 2] = 12 * E * I_y / L ** 3
        k_local[2, 4] = 6 * E * I_y / L ** 2
        k_local[2, 8] = -12 * E * I_y / L ** 3
        k_local[2, 10] = 6 * E * I_y / L ** 2
        k_local[4, 2] = 6 * E * I_y / L ** 2
        k_local[4, 4] = 4 * E * I_y / L
        k_local[4, 8] = -6 * E * I_y / L ** 2
        k_local[4, 10] = 2 * E * I_y / L
        k_local[8, 2] = -12 * E * I_y / L ** 3
        k_local[8, 4] = -6 * E * I_y / L ** 2
        k_local[8, 8] = 12 * E * I_y / L ** 3
        k_local[8, 10] = -6 * E * I_y / L ** 2
        k_local[10, 2] = 6 * E * I_y / L ** 2
        k_local[10, 4] = 2 * E * I_y / L
        k_local[10, 8] = -6 * E * I_y / L ** 2
        k_local[10, 10] = 4 * E * I_y / L
        k_local[1, 1] = 12 * E * I_z / L ** 3
        k_local[1, 5] = -6 * E * I_z / L ** 2
        k_local[1, 7] = -12 * E * I_z / L ** 3
        k_local[1, 11] = -6 * E * I_z / L ** 2
        k_local[5, 1] = -6 * E * I_z / L ** 2
        k_local[5, 5] = 4 * E * I_z / L
        k_local[5, 7] = 6 * E * I_z / L ** 2
        k_local[5, 11] = 2 * E * I_z / L
        k_local[7, 1] = -12 * E * I_z / L ** 3
        k_local[7, 5] = 6 * E * I_z / L ** 2
        k_local[7, 7] = 12 * E * I_z / L ** 3
        k_local[7, 11] = 6 * E * I_z / L ** 2
        k_local[11, 1] = -6 * E * I_z / L ** 2
        k_local[11, 5] = 2 * E * I_z / L
        k_local[11, 7] = 6 * E * I_z / L ** 2
        k_local[11, 11] = 4 * E * I_z / L
        k_local[3, 3] = G * J / L
        k_local[3, 9] = -G * J / L
        k_local[9, 3] = -G * J / L
        k_local[9, 9] = G * J / L
        k_global = T.T @ k_local @ T
        dofs = np.concatenate([np.arange(6 * node_i, 6 * node_i + 6), np.arange(6 * node_j, 6 * node_j + 6)])
        for i in range(12):
            for j in range(12):
                K[dofs[i], dofs[j]] += k_global[i, j]
    fixed_dofs = []
    free_dofs = []
    for i in range(N):
        if i in boundary_conditions:
            bc = boundary_conditions[i]
            for j in range(6):
                dof = 6 * i + j
                if bc[j] == 1:
                    fixed_dofs.append(dof)
                else:
                    free_dofs.append(dof)
        else:
            for j in range(6):
                free_dofs.append(6 * i + j)
    free_dofs = np.array(free_dofs)
    fixed_dofs = np.array(fixed_dofs)
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_fs = K[np.ix_(free_dofs, fixed_dofs)]
    K_sf = K[np.ix_(fixed_dofs, free_dofs)]
    K_ss = K[np.ix_(fixed_dofs, fixed_dofs)]
    P_f = P[free_dofs]
    P_s = P[fixed_dofs]
    if len(free_dofs) > 0:
        cond_num = np.linalg.cond(K_ff)
        if cond_num >= 1e+16:
            raise ValueError('Free-free stiffness submatrix is ill-conditioned')
        u_f = np.linalg.solve(K_ff, P_f)
    else:
        u_f = np.array([])
    u = np.zeros(ndof)
    if len(free_dofs) > 0:
        u[free_dofs] = u_f
    r = np.zeros(ndof)
    if len(fixed_dofs) > 0:
        if len(free_dofs) > 0:
            r_s = K_sf @ u_f - P_s
        else:
            r_s = -P_s
        r[fixed_dofs] = r_s
    return (u, r)