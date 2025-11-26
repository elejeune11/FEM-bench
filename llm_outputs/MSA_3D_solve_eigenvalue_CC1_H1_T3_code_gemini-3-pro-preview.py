def MSA_3D_solve_eigenvalue_CC1_H1_T3(K_e_global: np.ndarray, K_g_global: np.ndarray, boundary_conditions: dict, n_nodes: int):
    """
    Compute the smallest positive elastic critical load factor and corresponding
    global buckling mode shape for a 3D frame/beam model.
    The generalized eigenproblem is solved on the free DOFs:
        K_e_ff * phi = -lambda * K_g_ff * phi
    where K_e_ff and K_g_ff are the partitions of the elastic and geometric
    stiffness matrices after applying boundary conditions.
    Parameters
    ----------
    K_e_global : ndarray, shape (6*n_nodes, 6*n_nodes)
        Global elastic stiffness matrix.
    K_g_global : ndarray, shape (6*n_nodes, 6*n_nodes)
        Global geometric stiffness matrix at the reference load state.
    boundary_conditions : dict[int, array-like of bool]
        Dictionary mapping each node index (0-based) to a 6-element boolean array
        defining the constrained DOFs:
            True  → fixed (prescribed zero displacement/rotation)
            False → free (unknown)
        Nodes not listed are assumed fully free.
    n_nodes : int
        Number of nodes in the model (assumes 6 DOFs per node, ordered
        [u_x, u_y, u_z, theta_x, theta_y, theta_z] per node).
    Returns
    -------
    elastic_critical_load_factor : float
        The smallest positive eigenvalue lambda (> 0), interpreted as the elastic
        critical load factor (i.e., P_cr = lambda * P_ref, if K_g_global was
        formed at reference load P_ref).
    deformed_shape_vector : ndarray, shape (6*n_nodes,)
        Global buckling mode vector with entries on constrained DOFs set to zero.
        No normalization is applied (matches original behavior).
    Raises
    ------
    ValueError
            Use a tolerence of 1e16
    """
    from scipy import linalg
    total_dofs = 6 * n_nodes
    is_free = np.ones(total_dofs, dtype=bool)
    for (node_idx, constraints) in boundary_conditions.items():
        start_idx = node_idx * 6
        end_idx = start_idx + 6
        is_free[start_idx:end_idx] = ~np.array(constraints, dtype=bool)
    free_dofs_indices = np.nonzero(is_free)[0]
    if free_dofs_indices.size == 0:
        raise ValueError('Model has no free degrees of freedom.')
    idx_mesh = np.ix_(free_dofs_indices, free_dofs_indices)
    K_e_ff = K_e_global[idx_mesh]
    K_g_ff = K_g_global[idx_mesh]
    try:
        cond_num = np.linalg.cond(K_e_ff)
    except np.linalg.LinAlgError:
        raise ValueError('Elastic stiffness matrix is singular.')
    if cond_num > 1e+16:
        raise ValueError(f'Elastic stiffness matrix is ill-conditioned (cond={cond_num:.2e}).')
    try:
        (vals, vecs) = linalg.eig(K_e_ff, -K_g_ff)
    except linalg.LinAlgError:
        raise ValueError('Eigenvalue solver failed.')
    tol_complex = 1e-08
    if np.any(np.abs(vals.imag) > tol_complex) or np.any(np.abs(vecs.imag) > tol_complex):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    vals_real = vals.real
    vecs_real = vecs.real
    valid_mask = (vals_real > 1e-10) & np.isfinite(vals_real)
    valid_indices = np.nonzero(valid_mask)[0]
    if valid_indices.size == 0:
        raise ValueError('No positive elastic critical load factor found.')
    valid_vals = vals_real[valid_indices]
    min_idx_local = np.argmin(valid_vals)
    elastic_critical_load_factor = float(valid_vals[min_idx_local])
    target_idx = valid_indices[min_idx_local]
    reduced_mode_shape = vecs_real[:, target_idx]
    deformed_shape_vector = np.zeros(total_dofs, dtype=float)
    deformed_shape_vector[free_dofs_indices] = reduced_mode_shape
    return (elastic_critical_load_factor, deformed_shape_vector)