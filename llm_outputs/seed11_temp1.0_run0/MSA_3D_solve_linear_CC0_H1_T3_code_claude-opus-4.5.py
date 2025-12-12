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
    n_dofs = 6 * n_nodes
    fixed_dofs = np.zeros(n_dofs, dtype=bool)
    for (node_idx, constraints) in boundary_conditions.items():
        constraints = np.asarray(constraints, dtype=bool)
        start_dof = node_idx * 6
        for i in range(6):
            if constraints[i]:
                fixed_dofs[start_dof + i] = True
    fixed_indices = np.where(fixed_dofs)[0]
    free_indices = np.where(~fixed_dofs)[0]
    K_ff = K_global[np.ix_(free_indices, free_indices)]
    K_sf = K_global[np.ix_(fixed_indices, free_indices)]
    P_f = P_global[free_indices]
    P_s = P_global[fixed_indices]
    if len(free_indices) > 0:
        cond_num = np.linalg.cond(K_ff)
        if cond_num >= 1e+16:
            raise ValueError('The reduced stiffness matrix K_ff is ill-conditioned (cond(K_ff) >= 1e16), indicating a singular or unstable system.')
        u_free = np.linalg.solve(K_ff, P_f)
    else:
        u_free = np.array([])
    u = np.zeros(n_dofs)
    u[free_indices] = u_free
    r = np.zeros(n_dofs)
    if len(fixed_indices) > 0 and len(free_indices) > 0:
        r[fixed_indices] = K_sf @ u_free - P_s
    elif len(fixed_indices) > 0:
        r[fixed_indices] = -P_s
    return (u, r)