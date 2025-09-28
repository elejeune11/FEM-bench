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
    for (node_idx, loads) in nodal_loads.items():
        start_dof = 6 * node_idx
        P_global[start_dof:start_dof + 6] = loads
    fixed_dofs = np.zeros(total_dofs, dtype=bool)
    for (node_idx, bc) in boundary_conditions.items():
        start_dof = 6 * node_idx
        fixed_dofs[start_dof:start_dof + 6] = bc
    free_dofs = ~fixed_dofs

    def compute_element_stiffness(node_i, node_j, E, nu, A, I_y, I_z, J, local_z=None):
        coords_i = node_coords[node_i]
        coords_j = node_coords[node_j]
        L_vec = coords_j - coords_i
        L = np.linalg.norm(L_vec)
        if L == 0:
            raise ValueError('Zero-length element detected')
        local_x = L_vec / L
        if local_z is None:
            if abs(local_x[0]) > 1e-10 or abs(local_x[1]) > 1e-10:
                temp_vec = np.array([0, 0, 1.0])
            else:
                temp_vec = np.array([1.0, 0, 0])
            local_y = np.cross(temp_vec, local_x)
            local_y = local_y / np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
        else:
            local_z = np.array(local_z, dtype=float)
            local_z = local_z / np.linalg.norm(local_z)
            local_y = np.cross(local_z, local_x)
            local_y = local_y / np.linalg.norm(local_y)
            local_z = np.cross(local_x, local_y)
        T = np.zeros((3, 3))
        T[0, :] = local_x
        T[1, :] = local_y
        T[2, :] = local_z
        R = np.zeros((12, 12))
        for i in range(4):
            block_start = 3 * i
            R[block_start:block_start + 3, block_start:block_start + 3] = T
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
        k_local[2, 4] = 6 * L * EIy_L3
        k_local[2, 8] = -12 * EIy_L3
        k_local[2, 10] = 6 * L * EIy_L3
        k_local[4, 2] = 6 * L * EIy_L3
        k_local[4, 4] = 4 * L ** 2 * EIy_L3
        k_local[4, 8] = -6 * L * EIy_L3
        k_local[4, 10] = 2 * L ** 2 * EIy_L3
        k_local[8, 2] = -12 * EIy_L3
        k_local[8, 4] = -6 * L * EIy_L3
        k_local[8, 8] = 12 * EIy_L3
        k_local[8, 10] = -6 * L * EIy_L3
        k_local[10, 2] = 6 * L * EIy_L3
        k_local[10, 4] = 2 * L ** 2 * EIy_L3
        k_local[10, 8] = -6 * L * EIy_L3
        k_local[10, 10] = 4 * L ** 2 * EIy_L3
        EIz_L3 = E * I_z / L ** 3
        k_local[1, 1] = 12 * EIz_L3
        k_local[1, 5] = -6 * L * EIz_L3
        k_local[1, 7] = -12 * EIz_L3
        k_local[1, 11] = -6 * L * EIz_L3
        k_local[5, 1] = -6 * L * EIz_L3
        k_local[5, 5] = 4 * L ** 2 * EIz_L3
        k_local[5, 7] = 6 * L * EIz_L3
        k_local[5, 11] = 2 * L ** 2 * EIz_L3
        k_local[7, 1] = -12 * EIz_L3
        k_local[7, 5] = 6 * L * EIz_L3
        k_local[7, 7] = 12 * EIz_L3
        k_local[7, 11] = 6 * L * EIz_L3
        k_local[11, 1] = -6 * L * EIz_L3
        k_local[11, 5] = 2 * L ** 2 * EIz_L3
        k_local[11, 7] = 6 * L * EIz_L3
        k_local[11, 11] = 4 * L ** 2 * EIz_L3
        k_global = R.T @ k_local @ R
        return k_global
    for elem in elements:
        node_i = elem['node_i']
        node_j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        local_z = elem.get('local_z', None)
        k_elem = compute_element_stiffness(node_i, node_j, E, nu, A, I_y, I_z, J, local_z)
        dofs_i = np.arange(6 * node_i, 6 * node_i + 6)
        dofs_j = np.arange(6 * node_j, 6 * node_j + 6)
        assembly_dofs = np.concatenate([dofs_i, dofs_j])
        for (i, dof_i) in enumerate(assembly_dofs):
            for (j, dof_j) in enumerate(assembly_dofs):
                K_global[dof_i, dof_j] += k_elem[i, j]
    K_ff = K_global[free_dofs, :][:, free_dofs]
    K_sf = K_global[fixed_dofs, :][:, free_dofs]
    P_f = P_global[free_dofs]
    cond_num = np.linalg.cond(K_ff)
    if cond_num >= 1e+16:
        raise ValueError(f'Free-free stiffness matrix is ill-conditioned (cond = {cond_num:.2e})')
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(total_dofs)
    u[free_dofs] = u_f
    r = np.zeros(total_dofs)
    r[fixed_dofs] = K_global[fixed_dofs, :] @ u - P_global[fixed_dofs]
    return (u, r)