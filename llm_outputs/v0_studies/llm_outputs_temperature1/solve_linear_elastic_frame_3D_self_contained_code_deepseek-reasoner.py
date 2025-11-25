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
    for elem in elements:
        i = elem['node_i']
        j = elem['node_j']
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        coords_i = node_coords[i]
        coords_j = node_coords[j]
        L_vec = coords_j - coords_i
        L = np.linalg.norm(L_vec)
        if L < 1e-12:
            continue
        x_local = L_vec / L
        if elem.get('local_z') is not None:
            z_local = np.array(elem['local_z'])
            z_local = z_local / np.linalg.norm(z_local)
            if abs(np.dot(x_local, z_local)) > 0.99:
                raise ValueError('local_z is nearly parallel to beam axis')
            y_local = np.cross(z_local, x_local)
            y_local = y_local / np.linalg.norm(y_local)
            z_local = np.cross(x_local, y_local)
        else:
            if abs(x_local[2]) > 0.99:
                y_local = np.array([0.0, 1.0, 0.0])
            else:
                y_local = np.array([0.0, 0.0, 1.0])
            z_local = np.cross(x_local, y_local)
            z_local = z_local / np.linalg.norm(z_local)
            y_local = np.cross(z_local, x_local)
        T = np.vstack([x_local, y_local, z_local])
        R = np.zeros((12, 12))
        R[0:3, 0:3] = T
        R[3:6, 3:6] = T
        R[6:9, 6:9] = T
        R[9:12, 9:12] = T
        G = E / (2 * (1 + nu))
        k_local = np.zeros((12, 12))
        k_local[0, 0] = E * A / L
        k_local[0, 6] = -E * A / L
        k_local[6, 0] = -E * A / L
        k_local[6, 6] = E * A / L
        k_local[1, 1] = 12 * E * I_z / L ** 3
        k_local[1, 5] = 6 * E * I_z / L ** 2
        k_local[1, 7] = -12 * E * I_z / L ** 3
        k_local[1, 11] = 6 * E * I_z / L ** 2
        k_local[5, 1] = 6 * E * I_z / L ** 2
        k_local[5, 5] = 4 * E * I_z / L
        k_local[5, 7] = -6 * E * I_z / L ** 2
        k_local[5, 11] = 2 * E * I_z / L
        k_local[7, 1] = -12 * E * I_z / L ** 3
        k_local[7, 5] = -6 * E * I_z / L ** 2
        k_local[7, 7] = 12 * E * I_z / L ** 3
        k_local[7, 11] = -6 * E * I_z / L ** 2
        k_local[11, 1] = 6 * E * I_z / L ** 2
        k_local[11, 5] = 2 * E * I_z / L
        k_local[11, 7] = -6 * E * I_z / L ** 2
        k_local[11, 11] = 4 * E * I_z / L
        k_local[2, 2] = 12 * E * I_y / L ** 3
        k_local[2, 4] = -6 * E * I_y / L ** 2
        k_local[2, 8] = -12 * E * I_y / L ** 3
        k_local[2, 10] = -6 * E * I_y / L ** 2
        k_local[4, 2] = -6 * E * I_y / L ** 2
        k_local[4, 4] = 4 * E * I_y / L
        k_local[4, 8] = 6 * E * I_y / L ** 2
        k_local[4, 10] = 2 * E * I_y / L
        k_local[8, 2] = -12 * E * I_y / L ** 3
        k_local[8, 4] = 6 * E * I_y / L ** 2
        k_local[8, 8] = 12 * E * I_y / L ** 3
        k_local[8, 10] = 6 * E * I_y / L ** 2
        k_local[10, 2] = -6 * E * I_y / L ** 2
        k_local[10, 4] = 2 * E * I_y / L
        k_local[10, 8] = 6 * E * I_y / L ** 2
        k_local[10, 10] = 4 * E * I_y / L
        k_local[3, 3] = G * J / L
        k_local[3, 9] = -G * J / L
        k_local[9, 3] = -G * J / L
        k_local[9, 9] = G * J / L
        k_global_elem = R.T @ k_local @ R
        dofs_i = slice(6 * i, 6 * i + 6)
        dofs_j = slice(6 * j, 6 * j + 6)
        dofs = np.r_[dofs_i, dofs_j]
        K_global[np.ix_(dofs, dofs)] += k_global_elem
    for (node, loads) in nodal_loads.items():
        if 0 <= node < n_nodes:
            start_dof = 6 * node
            P_global[start_dof:start_dof + 6] = loads
    fixed_mask = np.zeros(n_dofs, dtype=bool)
    for (node, bc) in boundary_conditions.items():
        if 0 <= node < n_nodes:
            start_dof = 6 * node
            fixed_mask[start_dof:start_dof + 6] = bc
    free_mask = ~fixed_mask
    K_ff = K_global[free_mask][:, free_mask]
    K_fr = K_global[free_mask][:, fixed_mask]
    P_f = P_global[free_mask]
    cond_num = np.linalg.cond(K_ff)
    if cond_num >= 1e+16:
        raise ValueError(f'Ill-conditioned system: cond(K_ff) = {cond_num} >= 1e16')
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros(n_dofs)
    u[free_mask] = u_f
    r = np.zeros(n_dofs)
    r[fixed_mask] = K_global[fixed_mask] @ u - P_global[fixed_mask]
    return (u, r)