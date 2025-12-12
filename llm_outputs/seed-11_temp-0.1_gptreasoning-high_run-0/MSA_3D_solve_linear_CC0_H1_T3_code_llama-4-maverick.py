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
    fixed_dofs = np.zeros(6 * n_nodes, dtype=bool)
    for (node, constraints) in boundary_conditions.items():
        start_idx = node * 6
        fixed_dofs[start_idx:start_idx + 6] = constraints
    free_dofs = ~fixed_dofs
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_sf = K_global[np.ix_(fixed_dofs, free_dofs)]
    cond_K_ff = np.linalg.cond(K_ff)
    if cond_K_ff >= 1e+16:
        raise ValueError('The reduced stiffness matrix K_ff is ill-conditioned.')
    P_free = P_global[free_dofs]
    u_free = np.linalg.solve(K_ff, P_free)
    u = np.zeros_like(P_global)
    u[free_dofs] = u_free
    r = np.zeros_like(P_global)
    r[fixed_dofs] = K_sf @ u_free - P_global[fixed_dofs]
    return (u, r)