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
    dof_per_node = 6
    total_dof = dof_per_node * int(n_nodes)
    P_global = np.asarray(P_global, dtype=float).reshape(-1)
    K_global = np.asarray(K_global, dtype=float)
    is_fixed = np.zeros(total_dof, dtype=bool)
    if boundary_conditions is None:
        boundary_conditions = {}
    for node_idx, bc in boundary_conditions.items():
        if not 0 <= node_idx < n_nodes:
            continue
        bc_arr = np.asarray(bc, dtype=bool).reshape(-1)
        if bc_arr.size != dof_per_node:
            if bc_arr.size < dof_per_node:
                bc_arr = np.pad(bc_arr, (0, dof_per_node - bc_arr.size), constant_values=False)
            else:
                bc_arr = bc_arr[:dof_per_node]
        start = node_idx * dof_per_node
        is_fixed[start:start + dof_per_node] = bc_arr
    free_mask = ~is_fixed
    free_idx = np.nonzero(free_mask)[0]
    fixed_idx = np.nonzero(is_fixed)[0]
    u = np.zeros(total_dof, dtype=float)
    if free_idx.size > 0:
        K_ff = K_global[np.ix_(free_idx, free_idx)]
        cond = np.linalg.cond(K_ff)
        if not np.isfinite(cond) or cond >= 1e+16:
            raise ValueError('Reduced stiffness matrix K_ff is ill-conditioned (condition number >= 1e16).')
        P_f = P_global[free_idx]
        u_f = np.linalg.solve(K_ff, P_f)
        u[free_idx] = u_f
    r = np.zeros(total_dof, dtype=float)
    if fixed_idx.size > 0:
        if free_idx.size > 0:
            K_sf = K_global[np.ix_(fixed_idx, free_idx)]
            r_fixed = K_sf @ u[free_idx] - P_global[fixed_idx]
        else:
            r_fixed = -P_global[fixed_idx]
        r[fixed_idx] = r_fixed
    return (u, r)