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

    def get_dof_indices(node_idx):
        start = 6 * node_idx
        return slice(start, start + 6)
    for (node_idx, loads) in nodal_loads.items():
        dofs = get_dof_indices(node_idx)
        P_global[dofs] = loads
    fixed_dofs_mask = np.zeros(total_dofs, dtype=bool)
    for (node_idx, bc) in boundary_conditions.items():
        dofs = get_dof_indices(node_idx)
        fixed_dofs_mask[dofs] = bc
    free_dofs_mask = ~fixed_dofs_mask
    for elem in elements:
        i = elem['node_i']
        j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        local_z = elem.get('local_z')
        coords_i = node_coords[i]
        coords_j = node_coords[j]
        L_vec = coords_j - coords_i
        L = np.linalg.norm(L_vec)
        local_x = L_vec / L
        if local_z is None:
            if abs(local_x[0]) < 0.9:
                temp_vec = np.array([1.0, 0.0, 0.0])
            else:
                temp_vec = np.array([0.0, 1.0, 0.0])
            local_z = np.cross(local_x, temp_vec)
            local_z /= np.linalg.norm(local_z)
        local_z = local_z - np.dot(local_z, local_x) * local_x
        local_z /= np.linalg.norm(local_z)
        local_y = np.cross(local_z, local_x)
        local_y /= np.linalg.norm(local_y)
        T = np.vstack([local_x, local_y, local_z]).T
        R = np.zeros((12, 12))
        R[0:3, 0:3] = T
        R[3:6, 3:6] = T
        R[6:9, 6:9] = T
        R[9:12, 9:12] = T
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
        EIy_L3 = E * I_y / L ** 3
        k_local[2, 2] = 12 * EIy_L3
        k_local[2, 4] = 6 * E * I_y / L ** 2
        k_local[2, 8] = -12 * EIy_L3
        k_local[2, 10] = 6 * E * I_y / L ** 2
        k_local[4, 2] = 6 * E * I_y / L ** 2
        k_local[4, 4] = 4 * E * I_y / L
        k_local[4, 8] = -6 * E * I_y / L ** 2
        k_local[4, 10] = 2 * E * I_y / L
        k_local[8, 2] = -12 * EIy_L3
        k_local[8, 4] = -6 * E * I_y / L ** 2
        k_local[8, 8] = 12 * EIy_L3
        k_local[8, 10] = -6 * E * I_y / L ** 2
        k_local[10, 2] = 6 * E * I_y / L ** 2
        k_local[10, 4] = 2 * E * I_y / L
        k_local[10, 8] = -6 * E * I_y / L ** 2
        k_local[10, 10] = 4 * E * I_y / L
        EIz_L3 = E * I_z / L ** 3
        k_local[1, 1] = 12 * EIz_L3
        k_local[1, 5] = -6 * E * I_z / L ** 2
        k_local[1, 7] = -12 * EIz_L3
        k_local[1, 11] = -6 * E * I_z / L ** 2
        k_local[5, 1] = -6 * E * I_z / L ** 2
        k_local[5, 5] = 4 * E * I_z / L
        k_local[5, 7] = 6 * E * I_z / L ** 2
        k_local[5, 11] = 2 * E * I_z / L
        k_local[7, 1] = -12 * EIz_L3
        k_local[7, 5] = 6 * E * I_z / L ** 2
        k_local[7, 7] = 12 * EIz_L3
        k_local[7, 11] = 6 * E * I_z / L ** 2
        k_local[11, 1] = -6 * E * I_z / L ** 2
        k_local[11, 5] = 2 * E * I_z / L
        k_local[11, 7] = 6 * E * I_z / L ** 2
        k_local[11, 11] = 4 * E * I_z / L
        k_global_elem = R.T @ k_local @ R
        dofs_i = get_dof_indices(i)
        dofs_j = get_dof_indices(j)
        K_global[dofs_i, dofs_i] += k_global_elem[0:6, 0:6]
        K_global[dofs_i, dofs_j] += k_global_elem[0:6, 6:12]
        K_global[dofs_j, dofs_i] += k_global_elem[6:12, 0:6]
        K_global[dofs_j, dofs_j] += k_global_elem[6:12, 6:12]
    K_ff = K_global[free_dofs_mask, :][:, free_dofs_mask]
    K_fs = K_global[free_dofs_mask, :][:, fixed_dofs_mask]
    P_f = P_global[free_dofs_mask]
    cond_num = np.linalg.cond(K_ff)
    if cond_num >= 1e+16:
        raise ValueError(f'Ill-conditioned system: cond(K_ff) = {cond_num} >= 1e16')
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(total_dofs)
    u[free_dofs_mask] = u_f
    r = np.zeros(total_dofs)
    r[fixed_dofs_mask] = K_global[fixed_dofs_mask, :] @ u - P_global[fixed_dofs_mask]
    return (u, r)