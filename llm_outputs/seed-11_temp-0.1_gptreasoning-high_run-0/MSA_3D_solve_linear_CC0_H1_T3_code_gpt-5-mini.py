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
    total_dofs = 6 * int(n_nodes)
    P = np.asarray(P_global).reshape(-1).astype(float)
    if P.size != total_dofs:
        raise ValueError('P_global must have length 6*n_nodes')
    K = np.asarray(K_global, dtype=float)
    if K.shape != (total_dofs, total_dofs):
        raise ValueError('K_global must have shape (6*n_nodes, 6*n_nodes)')
    fixed_mask = np.zeros(total_dofs, dtype=bool)
    if boundary_conditions is None:
        boundary_conditions = {}
    for (node, bc) in boundary_conditions.items():
        try:
            node_idx = int(node)
        except Exception:
            raise ValueError('Boundary condition keys must be integer node indices')
        if node_idx < 0 or node_idx >= n_nodes:
            raise ValueError(f'Boundary condition node index {node_idx} out of range')
        arr = np.asarray(bc, dtype=bool).reshape(-1)
        if arr.size != 6:
            raise ValueError(f'Boundary condition for node {node_idx} must be length 6')
        fixed_mask[node_idx * 6:(node_idx + 1) * 6] = arr
    free_mask = ~fixed_mask
    free_idx = np.nonzero(free_mask)[0]
    fixed_idx = np.nonzero(fixed_mask)[0]
    u = np.zeros(total_dofs, dtype=float)
    r = np.zeros(total_dofs, dtype=float)
    if free_idx.size == 0:
        if fixed_idx.size > 0:
            P_fixed = P[fixed_idx]
            r[fixed_idx] = -P_fixed
        return (u, r)
    K_ff = K[np.ix_(free_idx, free_idx)]
    P_f = P[free_idx]
    cond_Kff = np.linalg.cond(K_ff)
    if cond_Kff >= 1e+16:
        raise ValueError('Reduced stiffness matrix K_ff is ill-conditioned (cond(K_ff) >= 1e16)')
    u_free = np.linalg.solve(K_ff, P_f)
    u[free_idx] = u_free
    if fixed_idx.size > 0:
        K_sf = K[np.ix_(fixed_idx, free_idx)]
        P_fixed = P[fixed_idx]
        r_fixed = K_sf.dot(u_free) - P_fixed
        r[fixed_idx] = r_fixed
    return (u, r)