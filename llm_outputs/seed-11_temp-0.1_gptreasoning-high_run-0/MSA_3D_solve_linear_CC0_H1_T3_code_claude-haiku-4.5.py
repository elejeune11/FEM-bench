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
    u = np.zeros(6 * n_nodes)
    r = np.zeros(6 * n_nodes)
    fixed_dofs = []
    free_dofs = []
    for node_idx in range(n_nodes):
        node_bc = boundary_conditions.get(node_idx, np.array([False] * 6))
        for local_dof in range(6):
            global_dof = 6 * node_idx + local_dof
            if node_bc[local_dof]:
                fixed_dofs.append(global_dof)
            else:
                free_dofs.append(global_dof)
    fixed_dofs = np.array(fixed_dofs, dtype=int)
    free_dofs = np.array(free_dofs, dtype=int)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_sf = K_global[np.ix_(fixed_dofs, free_dofs)]
    P_free = P_global[free_dofs]
    P_fixed = P_global[fixed_dofs]
    cond_K_ff = np.linalg.cond(K_ff)
    if cond_K_ff >= 1e+16:
        raise ValueError(f'Ill-conditioned stiffness matrix: cond(K_ff) = {cond_K_ff}')
    u_free = np.linalg.solve(K_ff, P_free)
    u[free_dofs] = u_free
    u[fixed_dofs] = 0.0
    r_fixed = K_sf @ u_free - P_fixed
    r[fixed_dofs] = r_fixed
    return (u, r)