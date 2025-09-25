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
    K_global = np.zeros((n_dofs, n_dofs))
    P_global = np.zeros(n_dofs)

    def get_dof_indices(node_idx):
        return slice(6 * node_idx, 6 * node_idx + 6)
    for (node_idx, loads) in nodal_loads.items():
        dofs = get_dof_indices(node_idx)
        P_global[dofs] = loads
    for element in elements:
        i = element['node_i']
        j = element['node_j']
        E = element['E']
        nu = element['nu']
        A = element['A']
        I_y = element['I_y']
        I_z = element['I_z']
        J = element['J']
        G = E / (2 * (1 + nu))
        coord_i = node_coords[i]
        coord_j = node_coords[j]
        L_vec = coord_j - coord_i
        L = np.linalg.norm(L_vec)
        e_x = L_vec / L
        if element['local_z'] is not None:
            local_z_ref = np.array(element['local_z'])
            e_z = local_z_ref - np.dot(local_z_ref, e_x) * e_x
            e_z = e_z / np.linalg.norm(e_z)
            e_y = np.cross(e_z, e_x)
        elif abs(np.dot(e_x, np.array([0, 0, 1]))) < 0.99:
            e_z = np.array([0, 0, 1])
            e_z = e_z - np.dot(e_z, e_x) * e_x
            e_z = e_z / np.linalg.norm(e_z)
            e_y = np.cross(e_z, e_x)
        else:
            e_y = np.array([0, 1, 0])
            e_y = e_y - np.dot(e_y, e_x) * e_x
            e_y = e_y / np.linalg.norm(e_y)
            e_z = np.cross(e_x, e_y)
        R = np.column_stack([e_x, e_y, e_z])
        T = np.zeros((12, 12))
        for block in range(4):
            start = 3 * block
            end = start + 3
            T[start:end, start:end] = R
        k_local = np.zeros((12, 12))
        EA_L = E * A / L
        k_local[0, 0] = EA_L
        k_local[6, 6] = EA_L
        k_local[0, 6] = -EA_L
        k_local[6, 0] = -EA_L
        GJ_L = G * J / L
        k_local[3, 3] = GJ_L
        k_local[9, 9] = GJ_L
        k_local[3, 9] = -GJ_L
        k_local[9, 3] = -GJ_L
        EIy_L = E * I_y / L
        EIy_L2 = E * I_y / L ** 2
        EIy_L3 = E * I_y / L ** 3
        k_local[2, 2] = 12 * EIy_L3
        k_local[8, 8] = 12 * EIy_L3
        k_local[2, 8] = -12 * EIy_L3
        k_local[8, 2] = -12 * EIy_L3
        k_local[4, 4] = 4 * EIy_L
        k_local[10, 10] = 4 * EIy_L
        k_local[4, 10] = 2 * EIy_L
        k_local[10, 4] = 2 * EIy_L
        k_local[2, 4] = 6 * EIy_L2
        k_local[4, 2] = 6 * EIy_L2
        k_local[2, 10] = 6 * EIy_L2
        k_local[10, 2] = 6 * EIy_L2
        k_local[8, 4] = -6 * EIy_L2
        k_local[4, 8] = -6 * EIy_L2
        k_local[8, 10] = -6 * EIy_L2
        k_local[10, 8] = -6 * EIy_L2
        EIz_L = E * I_z / L
        EIz_L2 = E * I_z / L ** 2
        EIz_L3 = E * I_z / L ** 3
        k_local[1, 1] = 12 * EIz_L3
        k_local[7, 7] = 12 * EIz_L3
        k_local[1, 7] = -12 * EIz_L3
        k_local[7, 1] = -12 * EIz_L3
        k_local[5, 5] = 4 * EIz_L
        k_local[11, 11] = 4 * EIz_L
        k_local[5, 11] = 2 * EIz_L
        k_local[11, 5] = 2 * EIz_L
        k_local[1, 5] = -6 * EIz_L2
        k_local[5, 1] = -6 * EIz_L2
        k_local[1, 11] = -6 * EIz_L2
        k_local[11, 1] = -6 * EIz_L2
        k_local[7, 5] = 6 * EIz_L2
        k_local[5, 7] = 6 * EIz_L2
        k_local[7, 11] = 6 * EIz_L2
        k_local[11, 7] = 6 * EIz_L2
        k_global_element = T @ k_local @ T.T
        dofs_i = get_dof_indices(i)
        dofs_j = get_dof_indices(j)
        element_dofs = np.concatenate([np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)])
        for (idx1, dof1) in enumerate(element_dofs):
            for (idx2, dof2) in enumerate(element_dofs):
                K_global[dof1, dof2] += k_global_element[idx1, idx2]
    fixed_dofs = np.zeros(n_dofs, dtype=bool)
    for (node_idx, bcs) in boundary_conditions.items():
        node_dofs = get_dof_indices(node_idx)
        for (i, bc) in enumerate(bcs):
            if bc == 1:
                fixed_dofs[node_dofs.start + i] = True
    free_dofs = ~fixed_dofs
    K_ff = K_global[free_dofs, :][:, free_dofs]
    K_fs = K_global[free_dofs, :][:, fixed_dofs]
    P_f = P_global[free_dofs]
    P_s = P_global[fixed_dofs]
    if np.linalg.cond(K_ff) >= 1e+16:
        raise ValueError('Free-free stiffness submatrix is ill-conditioned')
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(n_dofs)
    u[free_dofs] = u_f
    r = np.zeros(n_dofs)
    K_sf = K_global[fixed_dofs, :][:, free_dofs]
    r_s = K_sf @ u_f - P_s
    r[fixed_dofs] = r_s
    return (u, r)