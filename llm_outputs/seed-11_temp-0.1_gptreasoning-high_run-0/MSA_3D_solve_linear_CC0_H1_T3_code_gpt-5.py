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
    ndof = 6 * int(n_nodes)
    P = np.asarray(P_global, dtype=float).reshape(-1)
    if P.size != ndof:
        raise ValueError(f'P_global size {P.size} does not match expected size {ndof} (6*n_nodes).')
    K = np.asarray(K_global, dtype=float)
    if K.ndim != 2 or K.shape[0] != ndof or K.shape[1] != ndof:
        raise ValueError(f'K_global shape {K.shape} does not match expected shape ({ndof}, {ndof}).')
    fixed_mask = np.zeros(ndof, dtype=bool)
    if boundary_conditions is None:
        boundary_conditions = {}
    for node in range(n_nodes):
        bc = boundary_conditions.get(node, None)
        if bc is None:
            node_fixed = np.zeros(6, dtype=bool)
        else:
            node_fixed = np.asarray(bc, dtype=bool).reshape(-1)
            if node_fixed.size != 6:
                raise ValueError(f'Boundary condition for node {node} must have length 6; got {node_fixed.size}.')
        start = 6 * node
        fixed_mask[start:start + 6] = node_fixed
    free_mask = ~fixed_mask
    free_idx = np.nonzero(free_mask)[0]
    fixed_idx = np.nonzero(fixed_mask)[0]
    u = np.zeros(ndof, dtype=float)
    r = np.zeros(ndof, dtype=float)
    if free_idx.size > 0:
        K_ff = K[np.ix_(free_idx, free_idx)]
        cond_Kff = np.linalg.cond(K_ff)
        if not np.isfinite(cond_Kff) or cond_Kff >= 1e+16:
            raise ValueError('Reduced stiffness matrix K_ff is ill-conditioned (cond >= 1e16).')
        P_f = P[free_idx]
        try:
            u_f = np.linalg.solve(K_ff, P_f)
        except np.linalg.LinAlgError:
            raise ValueError('Reduced stiffness matrix K_ff is singular or not solvable.')
        u[free_idx] = u_f
    else:
        u_f = np.zeros(0, dtype=float)
    if fixed_idx.size > 0:
        K_sf = K[np.ix_(fixed_idx, free_idx)] if free_idx.size > 0 else np.zeros((fixed_idx.size, 0), dtype=float)
        P_s = P[fixed_idx]
        r_s = K_sf @ u_f - P_s
        r[fixed_idx] = r_s
    return (u, r)