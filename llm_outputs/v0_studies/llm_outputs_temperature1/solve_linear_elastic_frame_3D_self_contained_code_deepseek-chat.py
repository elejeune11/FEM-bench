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
    K_global = np.zeros((total_dofs, total_dofs))
    P_global = np.zeros(total_dofs)

    def get_node_dofs(node_idx):
        return slice(6 * node_idx, 6 * node_idx + 6)
    for (node_idx, loads) in nodal_loads.items():
        dofs = get_node_dofs(node_idx)
        P_global[dofs] = np.array(loads)
    for elem in elements:
        i = elem['node_i']
        j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        local_z = elem.get('local_z', None)
        coords_i = node_coords[i]
        coords_j = node_coords[j]
        L_vec = coords_j - coords_i
        L = np.linalg.norm(L_vec)
        local_x = L_vec / L
        if local_z is not None:
            local_z = np.array(local_z) / np.linalg.norm(local_z)
            if abs(np.dot(local_x, local_z)) > 0.99:
                raise ValueError('local_z is nearly parallel to beam axis')
            local_y = np.cross(local_z, local_x)
            local_y = local_y / np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
        else:
            if abs(np.dot(local_x, [0, 0, 1])) < 0.99:
                local_z = np.array([0, 0, 1])
            else:
                local_z = np.array([0, 1, 0])
            local_y = np.cross(local_z, local_x)
            local_y = local_y / np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
        R = np.vstack([local_x, local_y, local_z])
        T = np.zeros((12, 12))
        for k in range(4):
            block_start = 3 * k
            T[block_start:block_start + 3, block_start:block_start + 3] = R
        G = E / (2 * (1 + nu))
        k_axial = E * A / L
        k_torsion = G * J / L
        k_bend_y = 12 * E * I_z / L ** 3
        k_bend_z = 12 * E * I_y / L ** 3
        k_moment_y = 4 * E * I_y / L
        k_moment_z = 4 * E * I_z / L
        k_local = np.zeros((12, 12))
        k_local[0, 0] = k_axial
        k_local[0, 6] = -k_axial
        k_local[6, 0] = -k_axial
        k_local[6, 6] = k_axial
        k_local[3, 3] = k_torsion
        k_local[3, 9] = -k_torsion
        k_local[9, 3] = -k_torsion
        k_local[9, 9] = k_torsion
        k_local[1, 1] = k_bend_z
        k_local[1, 5] = 6 * E * I_y / L ** 2
        k_local[1, 7] = -k_bend_z
        k_local[1, 11] = 6 * E * I_y / L ** 2
        k_local[5, 1] = 6 * E * I_y / L ** 2
        k_local[5, 5] = k_moment_y
        k_local[5, 7] = -6 * E * I_y / L ** 2
        k_local[5, 11] = 2 * E * I_y / L
        k_local[7, 1] = -k_bend_z
        k_local[7, 5] = -6 * E * I_y / L ** 2
        k_local[7, 7] = k_bend_z
        k_local[7, 11] = -6 * E * I_y / L ** 2
        k_local[11, 1] = 6 * E * I_y / L ** 2
        k_local[11, 5] = 2 * E * I_y / L
        k_local[11, 7] = -6 * E * I_y / L ** 2
        k_local[11, 11] = k_moment_y
        k_local[2, 2] = k_bend_y
        k_local[2, 4] = -6 * E * I_z / L ** 2
        k_local[2, 8] = -k_bend_y
        k_local[2, 10] = -6 * E * I_z / L ** 2
        k_local[4, 2] = -6 * E * I_z / L ** 2
        k_local[4, 4] = k_moment_z
        k_local[4, 8] = 6 * E * I_z / L ** 2
        k_local[4, 10] = 2 * E * I_z / L
        k_local[8, 2] = -k_bend_y
        k_local[8, 4] = 6 * E * I_z / L ** 2
        k_local[8, 8] = k_bend_y
        k_local[8, 10] = 6 * E * I_z / L ** 2
        k_local[10, 2] = -6 * E * I_z / L ** 2
        k_local[10, 4] = 2 * E * I_z / L
        k_local[10, 8] = 6 * E * I_z / L ** 2
        k_local[10, 10] = k_moment_z
        k_local = (k_local + k_local.T) / 2
        k_global_elem = T.T @ k_local @ T
        dofs_i = get_node_dofs(i)
        dofs_j = get_node_dofs(j)
        K_global[np.ix_(dofs_i, dofs_i)] += k_global_elem[:6, :6]
        K_global[np.ix_(dofs_i, dofs_j)] += k_global_elem[:6, 6:]
        K_global[np.ix_(dofs_j, dofs_i)] += k_global_elem[6:, :6]
        K_global[np.ix_(dofs_j, dofs_j)] += k_global_elem[6:, 6:]
    fixed_dofs = []
    for node_idx in range(N):
        if node_idx in boundary_conditions:
            bc = boundary_conditions[node_idx]
            for dof_local in range(6):
                if bc[dof_local] == 1:
                    fixed_dofs.append(6 * node_idx + dof_local)
    fixed_dofs = np.array(fixed_dofs, dtype=int)
    free_dofs = np.setdiff1d(np.arange(total_dofs), fixed_dofs)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_fr = K_global[np.ix_(free_dofs, fixed_dofs)]
    P_f = P_global[free_dofs]
    cond_num = np.linalg.cond(K_ff)
    if cond_num >= 1e+16:
        raise ValueError(f'Free-free stiffness matrix is ill-conditioned (cond = {cond_num:.2e})')
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(total_dofs)
    u[free_dofs] = u_f
    r = np.zeros(total_dofs)
    if len(fixed_dofs) > 0:
        K_rf = K_global[np.ix_(fixed_dofs, free_dofs)]
        r_fixed = K_rf @ u_f
        r[fixed_dofs] = r_fixed
    return (u, r)