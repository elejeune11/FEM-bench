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
    import numpy as np
    N = 6 * int(n_nodes)
    P = np.asarray(P_global, dtype=float).reshape(-1)
    K = np.asarray(K_global, dtype=float)
    if P.size != N:
        raise ValueError(f'P_global must have length {N}, got {P.size}')
    if K.shape != (N, N):
        raise ValueError(f'K_global must have shape ({N}, {N}), got {K.shape}')
    fixed_mask = np.zeros(N, dtype=bool)
    bc = {} if boundary_conditions is None else boundary_conditions
    for node in range(n_nodes):
        if node in bc:
            bc_node = np.asarray(bc[node], dtype=bool).reshape(-1)
            if bc_node.size != 6:
                raise ValueError(f'Boundary condition for node {node} must have 6 boolean entries.')
            start = node * 6
            fixed_mask[start:start + 6] = bc_node
    free_mask = ~fixed_mask
    free_idx = np.nonzero(free_mask)[0]
    fixed_idx = np.nonzero(fixed_mask)[0]
    u = np.zeros(N, dtype=float)
    r = np.zeros(N, dtype=float)
    if free_idx.size == 0:
        if fixed_idx.size > 0:
            r[fixed_idx] = -P[fixed_idx]
        return (u, r)
    K_ff = K[np.ix_(free_idx, free_idx)]
    condK = np.linalg.cond(K_ff)
    if not np.isfinite(condK) or condK >= 1e+16:
        raise ValueError('The reduced stiffness matrix K_ff is ill-conditioned (cond(K_ff) ≥ 1e16).')
    P_f = P[free_idx]
    u_f = np.linalg.solve(K_ff, P_f)
    u[free_idx] = u_f
    if fixed_idx.size > 0:
        K_cf = K[np.ix_(fixed_idx, free_idx)]
        P_c = P[fixed_idx]
        r_c = K_cf @ u_f - P_c
        r[fixed_idx] = r_c
    return (u, r)