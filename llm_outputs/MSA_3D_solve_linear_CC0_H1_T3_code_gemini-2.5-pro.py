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
    Helper Functions
    ----------------
        Identifies which global DOFs are fixed and which are free, returning
        sorted integer index arrays (`fixed`, `free`). This helper ensures
        consistency between the nodal boundary-condition specification and the
        DOF layout assumed here.
    Notes
    -----
      at fixed DOFs.
      for the solution to be valid.
    """
    n_dofs = 6 * n_nodes
    is_fixed_mask = np.zeros(n_dofs, dtype=bool)
    for (node_idx, constraints) in boundary_conditions.items():
        start_dof = node_idx * 6
        end_dof = start_dof + 6
        is_fixed_mask[start_dof:end_dof] = constraints
    free_dofs = np.where(~is_fixed_mask)[0]
    fixed_dofs = np.where(is_fixed_mask)[0]
    P_f = P_global[free_dofs]
    P_s = P_global[fixed_dofs]
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_sf = K_global[np.ix_(fixed_dofs, free_dofs)]
    if K_ff.shape[0] > 0:
        condition_number = np.linalg.cond(K_ff)
        if condition_number >= 1e+16:
            raise ValueError('Reduced stiffness matrix K_ff is ill-conditioned (cond >= 1e16), indicating a singular or unstable system.')
    u_f = np.linalg.solve(K_ff, P_f)
    r_s = K_sf @ u_f - P_s
    u = np.zeros(n_dofs)
    r = np.zeros(n_dofs)
    u[free_dofs] = u_f
    r[fixed_dofs] = r_s
    return (u, r)