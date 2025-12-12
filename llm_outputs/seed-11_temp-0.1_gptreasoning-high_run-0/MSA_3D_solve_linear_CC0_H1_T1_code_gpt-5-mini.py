def MSA_3D_solve_linear_CC0_H1_T1(P_global: np.ndarray, K_global: np.ndarray, boundary_conditions: dict, n_nodes: int):
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
    P = np.asarray(P_global, dtype=float).reshape(-1)
    n_dof = int(n_nodes) * 6
    if P.size != n_dof:
        raise ValueError('P_global length does not match number of DOFs')
    K = np.asarray(K_global, dtype=float)
    if K.shape != (n_dof, n_dof):
        raise ValueError('K_global shape does not match number of DOFs')
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    fixed = np.asarray(fixed, dtype=int)
    free = np.asarray(free, dtype=int)
    u = np.zeros(n_dof, dtype=float)
    r = np.zeros(n_dof, dtype=float)
    if free.size > 0:
        K_ff = K[np.ix_(free, free)]
        P_f = P[free]
        cond = np.linalg.cond(K_ff)
        if not np.isfinite(cond) or cond >= 1e+16:
            raise ValueError('Reduced stiffness matrix K_ff is ill-conditioned (cond >= 1e16)')
        try:
            u_f = np.linalg.solve(K_ff, P_f)
        except np.linalg.LinAlgError as exc:
            raise ValueError('Reduced stiffness matrix K_ff is singular or ill-conditioned') from exc
        u[free] = u_f
    if fixed.size > 0:
        if free.size > 0:
            K_cf = K[np.ix_(fixed, free)]
            r_fixed = K_cf.dot(u[free]) - P[fixed]
        else:
            r_fixed = -P[fixed]
        r[fixed] = r_fixed
    return (u, r)