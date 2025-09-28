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
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K = np.zeros((n_dofs, n_dofs))
    P = np.zeros(n_dofs)
    for (node_idx, load) in nodal_loads.items():
        start_dof = 6 * node_idx
        P[start_dof:start_dof + 6] = load
    for elem in elements:
        i = elem['node_i']
        j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        local_z_ref = elem.get('local_z')
        coord_i = node_coords[i]
        coord_j = node_coords[j]
        L_vec = coord_j - coord_i
        L = np.linalg.norm(L_vec)
        e_x = L_vec / L
        if local_z_ref is not None:
            e_z = np.array(local_z_ref, dtype=float)
            e_z = e_z / np.linalg.norm(e_z)
            if abs(np.dot(e_x, e_z)) > 0.99:
                raise ValueError('local_z vector is nearly parallel to beam axis')
            e_y = np.cross(e_z, e_x)
            e_y = e_y / np.linalg.norm(e_y)
            e_z = np.cross(e_x, e_y)
        else:
            if abs(e_x[2]) < 1e-10:
                e_z = np.array([0, 0, 1])
            else:
                proj_y = np.array([-e_x[1], e_x[0], 0])
                if np.linalg.norm(proj_y) < 1e-10:
                    proj_y = np.array([1, 0, 0])
                e_y = proj_y / np.linalg.norm(proj_y)
            e_z = np.cross(e_x, e_y)
            e_z = e_z / np.linalg.norm(e_z)
            e_y = np.cross(e_z, e_x)
        T = np.column_stack([e_x, e_y, e_z])
        T_lg = np.zeros((12, 12))
        for block in range(4):
            start = 3 * block
            T_lg[start:start + 3, start:start + 3] = T
        G = E / (2 * (1 + nu))
        k_local = np.zeros((12, 12))
        EA_L = E * A / L
        k_local[0, 0] = EA_L
        k_local[0, 6] = -EA_L
        k_local[6, 0] = -EA_L
        k_local[6, 6] = EA_L
        GJ_L = G * J / L
        k_local[3, 3] = GJ_L
        k_local[3, 9] = -GJ_L
        k_local[9, 3] = -GJ_L
        k_local[9, 9] = GJ_L
        EIy_L = E * I_y / L
        EIy_L2 = E * I_y / L ** 2
        EIy_L3 = E * I_y / L ** 3
        k_local[[2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 10, 10, 10, 10], [2, 4, 8, 10, 2, 4, 8, 10, 2, 4, 8, 10, 2, 4, 8, 10]] = [12 * EIy_L3, 6 * EIy_L2, -12 * EIy_L3, 6 * EIy_L2, 6 * EIy_L2, 4 * EIy_L, -6 * EIy_L2, 2 * EIy_L, -12 * EIy_L3, -6 * EIy_L2, 12 * EIy_L3, -6 * EIy_L2, 6 * EIy_L2, 2 * EIy_L, -6 * EIy_L2, 4 * EIy_L]
        EIz_L = E * I_z / L
        EIz_L2 = E * I_z / L ** 2
        EIz_L3 = E * I_z / L ** 3
        k_local[[1, 1, 1, 1, 5, 5, 5, 5, 7, 7, 7, 7, 11, 11, 11, 11], [1, 5, 7, 11, 1, 5, 7, 11, 1, 5, 7, 11, 1, 5, 7, 11]] = [12 * EIz_L3, -6 * EIz_L2, -12 * EIz_L3, -6 * EIz_L2, -6 * EIz_L2, 4 * EIz_L, 6 * EIz_L2, 2 * EIz_L, -12 * EIz_L3, 6 * EIz_L2, 12 * EIz_L3, 6 * EIz_L2, -6 * EIz_L2, 2 * EIz_L, 6 * EIz_L2, 4 * EIz_L]
        k_global = T_lg @ k_local @ T_lg.T
        dofs_i = range(6 * i, 6 * i + 6)
        dofs_j = range(6 * j, 6 * j + 6)
        assembly_dofs = list(dofs_i) + list(dofs_j)
        for (idx_row, dof_row) in enumerate(assembly_dofs):
            for (idx_col, dof_col) in enumerate(assembly_dofs):
                K[dof_row, dof_col] += k_global[idx_row, idx_col]
    fixed_mask = np.zeros(n_dofs, dtype=bool)
    for (node_idx, bc) in boundary_conditions.items():
        start_dof = 6 * node_idx
        for (i, fixed) in enumerate(bc):
            fixed_mask[start_dof + i] = bool(fixed)
    free_mask = ~fixed_mask
    K_ff = K[free_mask][:, free_mask]
    K_fs = K[free_mask][:, fixed_mask]
    P_f = P[free_mask]
    if np.linalg.cond(K_ff) >= 1e+16:
        raise ValueError('Free-free stiffness matrix is ill-conditioned')
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(n_dofs)
    u[free_mask] = u_f
    r = K @ u - P
    r[free_mask] = 0.0
    return (u, r)