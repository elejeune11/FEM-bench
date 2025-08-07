def solve_linear_elastic_frame_3D(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    """
    Small-displacement linear-elastic analysis of a 3D frame. Coordinate system follows the right hand rule.
    The condition number of the global stiffness matrix should be checked before solving.
    If the problem is ill-posed based on condition number, return a (6 N, ) zero array for both u and r.
    Parameters
    ----------
    node_coords : (N, 3) float ndarray
        Global coordinates of the N nodes (row i → [x, y, z] of node i, 0-based).
    elements : iterable of dict
        Each dict must contain
            'node_i', 'node_j' : int # end node indices (0-based)
            'E', 'nu', 'A', 'I_y', 'I_z', 'J' : float
            'local_z' : (3,) array | None # optional unit vector for local z
    boundary_conditions : dict[int, Sequence[int]]
        node index → 6-element 0/1 iterable (0 = free, 1 = fixed).
        Omitted nodes ⇒ all DOFs free.
    nodal_loads : dict[int, Sequence[float]]
        node index → 6-element [Fx, Fy, Fz, Mx, My, Mz] (forces (+) and moments).
        Omitted nodes ⇒ zero loads.
    Returns:
    u : (6 N,) ndarray
        Global displacement vector (UX, UY, UZ, RX, RY, RZ for each node in order).
    r : (6 N,) ndarray
        Global force/moment vector with support reactions filled in fixed DOFs.
    """
    n_nodes = len(node_coords)
    ndof = 6 * n_nodes
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)
    for (node, loads) in nodal_loads.items():
        F[6 * node:6 * node + 6] = loads
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        dx = node_coords[j] - node_coords[i]
        L = np.linalg.norm(dx)
        (E, A) = (elem['E'], elem['A'])
        (Iy, Iz) = (elem['I_y'], elem['I_z'])
        J = elem['J']
        x_vec = dx / L
        if elem.get('local_z') is not None:
            z_vec = elem['local_z']
            y_vec = np.cross(z_vec, x_vec)
        elif abs(x_vec[2]) < 0.999:
            y_vec = np.cross([0, 0, 1], x_vec)
        else:
            y_vec = np.cross([0, 1, 0], x_vec)
        y_vec = y_vec / np.linalg.norm(y_vec)
        z_vec = np.cross(x_vec, y_vec)
        T = np.zeros((12, 12))
        R = np.vstack((x_vec, y_vec, z_vec))
        for k in range(4):
            T[3 * k:3 * k + 3, 3 * k:3 * k + 3] = R
        k = np.zeros((12, 12))
        EA_L = E * A / L
        k[0, 0] = k[6, 6] = EA_L
        k[0, 6] = k[6, 0] = -EA_L
        GJ_L = E * J / (2 * (1 + elem['nu'])) / L
        k[3, 3] = k[9, 9] = GJ_L
        k[3, 9] = k[9, 3] = -GJ_L
        EIy_L3 = E * Iy / L ** 3
        EIz_L3 = E * Iz / L ** 3
        k[1, 1] = k[7, 7] = 12 * EIz_L3
        k[1, 7] = k[7, 1] = -12 * EIz_L3
        k[1, 5] = k[5, 1] = 6 * L * EIz_L3
        k[1, 11] = k[11, 1] = 6 * L * EIz_L3
        k[5, 5] = k[11, 11] = 4 * L ** 2 * EIz_L3
        k[5, 7] = k[7, 5] = -6 * L * EIz_L3
        k[5, 11] = k[11, 5] = 2 * L ** 2 * EIz_L3
        k[7, 11] = k[11, 7] = -6 * L * EIz_L3
        k[2, 2] = k[8, 8] = 12 * EIy_L3
        k[2, 8] = k[8, 2] = -12 * EIy_L3
        k[2, 4] = k[4, 2] = -6 * L * EIy_L3
        k[2, 10] = k[10, 2] = -6 * L * EIy_L3
        k[4, 4] = k[10, 10] = 4 * L ** 2 * EIy_L3
        k[4, 8] = k[8, 4] = 6 * L * EIy_L3
        k[4, 10] = k[10, 4] = 2 * L ** 2 * EIy_L3
        k[8, 10] = k[10, 8] = 6 * L * EIy_L3
        k_global = T.T @ k @ T
        dofs_i = slice(6 * i, 6 * i + 6)
        dofs_j = slice(6 * j, 6 * j + 6)
        idx = np.r_[dofs_i, dofs_j]
        K[np.ix_(idx, idx)] += k_global
    free_dofs = np.ones(ndof, dtype=bool)
    for (node, fixity) in boundary_conditions.items():
        for (dof, is_fixed) in enumerate(fixity):
            if is_fixed:
                free_dofs[6 * node + dof] = False
    K_free = K[np.ix_(free_dofs, free_dofs)]
    if np.linalg.cond(K_free) > 1000000000000.0:
        return (np.zeros(ndof), np.zeros(ndof))
    u = np.zeros(ndof)
    u[free_dofs] = np.linalg.solve(K_free, F[free_dofs])
    r = K @ u
    return (u, r)