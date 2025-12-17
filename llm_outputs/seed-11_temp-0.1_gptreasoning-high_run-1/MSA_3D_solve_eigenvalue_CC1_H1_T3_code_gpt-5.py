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
    if not isinstance(n_nodes, int) or n_nodes <= 0:
        raise ValueError('n_nodes must be a positive integer.')
    total_dofs = 6 * n_nodes
    K_e = np.asarray(K_e_global)
    K_g = np.asarray(K_g_global)
    if K_e.shape != (total_dofs, total_dofs) or K_g.shape != (total_dofs, total_dofs):
        raise ValueError('Input matrices must have shape (6*n_nodes, 6*n_nodes).')
    fixed_mask = np.zeros(total_dofs, dtype=bool)
    if boundary_conditions is not None:
        for node_idx, bc in boundary_conditions.items():
            if node_idx < 0 or node_idx >= n_nodes:
                raise ValueError('Boundary condition node index out of range.')
            bc_arr = np.asarray(bc, dtype=bool).reshape(-1)
            if bc_arr.size != 6:
                raise ValueError('Each boundary condition must be a 6-element array-like.')
            start = 6 * node_idx
            fixed_mask[start:start + 6] = np.logical_or(fixed_mask[start:start + 6], bc_arr)
    free_mask = ~fixed_mask
    free_idx = np.flatnonzero(free_mask)
    if free_idx.size == 0:
        raise ValueError('All DOFs are constrained; no free DOFs to solve eigenproblem.')
    K_e_ff = K_e[np.ix_(free_idx, free_idx)]
    K_g_ff = K_g[np.ix_(free_idx, free_idx)]
    tol_cond = 1e+16
    try:
        cond_e = np.linalg.cond(K_e_ff)
    except np.linalg.LinAlgError:
        cond_e = np.inf
    try:
        cond_g = np.linalg.cond(K_g_ff)
    except np.linalg.LinAlgError:
        cond_g = np.inf
    if not np.isfinite(cond_e) or cond_e > tol_cond:
        raise ValueError('Reduced elastic stiffness matrix is ill-conditioned/singular beyond tolerance.')
    if not np.isfinite(cond_g) or cond_g > tol_cond:
        raise ValueError('Reduced geometric stiffness matrix is ill-conditioned/singular beyond tolerance.')
    try:
        w, v = scipy.linalg.eig(K_e_ff, -K_g_ff, check_finite=True)
    except Exception as e:
        raise ValueError(f'Generalized eigenvalue computation failed: {e}')
    w = np.asarray(w)
    imag_tol = 1e-08
    is_finite = np.isfinite(w)
    real_part = w.real
    imag_part = w.imag
    is_nearly_real = np.abs(imag_part) <= imag_tol * (1.0 + np.abs(real_part))
    positive = real_part > 0.0
    candidate_mask = is_finite & is_nearly_real & positive
    if not np.any(candidate_mask):
        raise ValueError('No positive eigenvalue is found.')
    candidate_indices = np.flatnonzero(candidate_mask)
    candidate_reals = real_part[candidate_indices]
    min_idx_local = int(np.argmin(candidate_reals))
    chosen_idx = int(candidate_indices[min_idx_local])
    chosen_lambda = float(candidate_reals[min_idx_local])
    vec = v[:, chosen_idx]
    if np.max(np.abs(vec.imag)) > imag_tol * (1.0 + np.max(np.abs(vec.real))):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    phi_free = np.real(vec)
    deformed_shape_vector = np.zeros(total_dofs, dtype=float)
    deformed_shape_vector[free_idx] = phi_free
    return (chosen_lambda, deformed_shape_vector)