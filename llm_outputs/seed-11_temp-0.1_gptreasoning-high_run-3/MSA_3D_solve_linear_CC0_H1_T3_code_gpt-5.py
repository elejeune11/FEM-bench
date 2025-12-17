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
    n_dofs = dof_per_node * int(n_nodes)
    P = np.asarray(P_global, dtype=float).reshape(-1)
    if P.size != n_dofs:
        raise ValueError(f'P_global must have length {n_dofs}, got {P.size}')
    K = np.asarray(K_global, dtype=float)
    if K.shape != (n_dofs, n_dofs):
        raise ValueError(f'K_global must have shape ({n_dofs}, {n_dofs}), got {K.shape}')
    fixed_mask = np.zeros(n_dofs, dtype=bool)
    if boundary_conditions is not None:
        if not isinstance(boundary_conditions, dict):
            raise ValueError('boundary_conditions must be a dict mapping node index to 6-element boolean array')
        for node, bc in boundary_conditions.items():
            if not isinstance(node, int):
                raise ValueError('Node indices in boundary_conditions must be integers')
            if node < 0 or node >= n_nodes:
                raise ValueError(f'Node index {node} out of range [0, {n_nodes - 1}]')
            arr = np.asarray(bc, dtype=bool).reshape(-1)
            if arr.size != dof_per_node:
                raise ValueError(f'Boundary condition for node {node} must have 6 elements, got {arr.size}')
            start = node * dof_per_node
            fixed_mask[start:start + dof_per_node] = arr
    free_idx = np.nonzero(~fixed_mask)[0]
    fixed_idx = np.nonzero(fixed_mask)[0]
    u = np.zeros(n_dofs, dtype=float)
    r = np.zeros(n_dofs, dtype=float)
    if free_idx.size > 0:
        K_ff = K[np.ix_(free_idx, free_idx)]
        P_f = P[free_idx]
        cond = np.linalg.cond(K_ff) if K_ff.size > 0 else 0.0
        if not np.isfinite(cond) or cond >= 1e+16:
            raise ValueError('Reduced stiffness matrix K_ff is ill-conditioned (cond(K_ff) >= 1e16)')
        u_f = np.linalg.solve(K_ff, P_f)
        u[free_idx] = u_f
        if fixed_idx.size > 0:
            K_sf = K[np.ix_(fixed_idx, free_idx)]
            P_s = P[fixed_idx]
            r_s = K_sf @ u_f - P_s
            r[fixed_idx] = r_s
    elif fixed_idx.size > 0:
        P_s = P[fixed_idx]
        r[fixed_idx] = -P_s
    return (u, r)