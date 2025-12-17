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
    ndof = 6 * n_nodes
    P = np.asarray(P_global, dtype=float).reshape(-1)
    K = np.asarray(K_global, dtype=float)
    if P.size != ndof:
        raise ValueError(f'P_global must have length {ndof}, got {P.size}')
    if K.shape != (ndof, ndof):
        raise ValueError(f'K_global must have shape ({ndof}, {ndof}), got {K.shape}')
    fixed, free = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    u = np.zeros(ndof, dtype=float)
    r = np.zeros(ndof, dtype=float)
    if free.size > 0:
        K_ff = K[np.ix_(free, free)]
        P_f = P[free]
        cond_Kff = np.linalg.cond(K_ff)
        if not np.isfinite(cond_Kff) or cond_Kff >= 1e+16:
            raise ValueError('Reduced stiffness matrix K_ff is ill-conditioned (cond(K_ff) ≥ 1e16); system may be singular or unstable.')
        u_free = np.linalg.solve(K_ff, P_f)
        u[free] = u_free
        if fixed.size > 0:
            K_sf = K[np.ix_(fixed, free)]
            P_s = P[fixed]
            r_fixed = K_sf @ u_free - P_s
            r[fixed] = r_fixed
    elif fixed.size > 0:
        r[fixed] = -P[fixed]
    return (u, r)