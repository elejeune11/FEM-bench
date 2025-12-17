def MSA_3D_solve_linear_CC0_H1_T3(P_global: np.ndarray, K_global: np.ndarray, boundary_conditions: dict, n_nodes: int):
    """
    Solve for nodal displacements and support reactions in a 3D linear-elastic frame
    using a partitioned stiffness approach.
        K_global * u_global = P_global
    into fixed and free degree-of-freedom (DOF) subsets based on the specified
    boundary conditions. The reduced system for the free DOFs is solved directly,
    provided that the free–free stiffness submatrix (K_ff) is well-conditioned.
    Reactions at fixed supports are then computed from the recovered free displacements.
    Parameters
    ----------
    P_global : (6*n_nodes,) ndarray of float
        Global load vector containing externally applied nodal forces and moments.
        Entries follow the per-node DOF order:
        [F_x, F_y, F_z, M_x, M_y, M_z].
    K_global : (6*n_nodes, 6*n_nodes) ndarray of float
        Assembled global stiffness matrix for the structure.
    boundary_conditions : dict[int, array-like of bool]
        Dictionary mapping each node index (0-based) to a 6-element boolean array
        defining the constrained DOFs:
            True  → fixed (prescribed zero displacement/rotation)
            False → free (unknown)
        Nodes not listed are assumed fully free.
    n_nodes : int
        Total number of nodes in the structure.
    Returns
    -------
    u : (6*n_nodes,) ndarray of float
        Global displacement vector. Free DOFs contain computed displacements;
        fixed DOFs are zero.
    r : (6*n_nodes,) ndarray of float
        Global reaction vector. Nonzero values appear only at fixed DOFs and
        represent internal support reactions:
            r_fixed = K_sf @ u_free - P_fixed.
    Raises
    ------
    ValueError
        If the reduced stiffness matrix K_ff is ill-conditioned
        (cond(K_ff) ≥ 1e16), indicating a singular or unstable system.
    Notes
    -----
      at fixed DOFs.
      for the solution to be valid.
    """
    if boundary_conditions is None:
        boundary_conditions = {}
    ndof = 6 * int(n_nodes)
    P = np.asarray(P_global, dtype=float).reshape(-1)
    K = np.asarray(K_global, dtype=float)
    is_fixed = np.zeros(ndof, dtype=bool)
    for node in range(n_nodes):
        bc_i = boundary_conditions.get(node, None)
        if bc_i is None:
            continue
        bc_arr = np.asarray(bc_i, dtype=bool).reshape(-1)
        bc6 = np.zeros(6, dtype=bool)
        n_copy = 6 if bc_arr.size >= 6 else bc_arr.size
        if n_copy > 0:
            bc6[:n_copy] = bc_arr[:n_copy]
        start = 6 * node
        is_fixed[start:start + 6] = bc6
    free_mask = ~is_fixed
    free_idx = np.flatnonzero(free_mask)
    fixed_idx = np.flatnonzero(is_fixed)
    u = np.zeros(ndof, dtype=float)
    r = np.zeros(ndof, dtype=float)
    if free_idx.size > 0:
        K_ff = K[np.ix_(free_idx, free_idx)]
        cond_num = np.linalg.cond(K_ff)
        if not np.isfinite(cond_num) or cond_num >= 1e+16:
            raise ValueError('Ill-conditioned reduced stiffness matrix K_ff (cond >= 1e16).')
        P_f = P[free_idx]
        u_f = np.linalg.solve(K_ff, P_f)
        u[free_idx] = u_f
    if fixed_idx.size > 0:
        K_sf = K[np.ix_(fixed_idx, free_idx)]
        r_fixed = K_sf @ u[free_idx] - P[fixed_idx]
        r[fixed_idx] = r_fixed
    return (u, r)