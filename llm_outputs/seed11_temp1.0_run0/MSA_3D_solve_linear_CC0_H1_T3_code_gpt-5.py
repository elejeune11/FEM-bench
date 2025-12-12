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
    dof = 6 * n_nodes
    fixed_mask = np.zeros(dof, dtype=bool)
    for (node, bc) in boundary_conditions.items():
        if node is None:
            continue
        if node < 0 or node >= n_nodes:
            continue
        bc_arr = np.asarray(bc, dtype=bool).reshape(-1)
        if bc_arr.size != 6:
            continue
        start = 6 * node
        fixed_mask[start:start + 6] = bc_arr
    free_mask = ~fixed_mask
    u = np.zeros(dof, dtype=float)
    r = np.zeros(dof, dtype=float)
    if np.any(free_mask):
        K_ff = K_global[np.ix_(free_mask, free_mask)]
        P_f = P_global[free_mask]
        cond_val = np.linalg.cond(K_ff)
        if not np.isfinite(cond_val) or cond_val >= 1e+16:
            raise ValueError('Reduced stiffness matrix K_ff is ill-conditioned (cond >= 1e16).')
        u_free = np.linalg.solve(K_ff, P_f)
        u[free_mask] = u_free
    else:
        u_free = np.zeros(0, dtype=float)
    if np.any(fixed_mask):
        K_sf = K_global[np.ix_(fixed_mask, free_mask)]
        P_s = P_global[fixed_mask]
        r_fixed = K_sf @ u_free - P_s
        r[fixed_mask] = r_fixed
    return (u, r)