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
    ndof = 6 * int(n_nodes)
    is_free = np.ones(ndof, dtype=bool)
    if boundary_conditions is not None:
        for node_idx, bc in boundary_conditions.items():
            base = 6 * int(node_idx)
            end = base + 6
            bc_arr = np.asarray(bc, dtype=bool).reshape(-1)
            is_free[base:end] = ~bc_arr[:6]
    free_idx = np.flatnonzero(is_free)
    if free_idx.size == 0:
        raise ValueError('Reduced matrices are ill-conditioned/singular beyond tolerance (no free DOFs).')
    K_e = np.asarray(K_e_global)
    K_g = np.asarray(K_g_global)
    K_e_ff = K_e[np.ix_(free_idx, free_idx)]
    K_g_ff = K_g[np.ix_(free_idx, free_idx)]
    tol_cond = 1e+16
    try:
        cond_e = np.linalg.cond(K_e_ff)
    except Exception:
        cond_e = np.inf
    try:
        cond_g = np.linalg.cond(K_g_ff)
    except Exception:
        cond_g = np.inf
    if not np.isfinite(cond_e) or cond_e > tol_cond or (not np.isfinite(cond_g)) or (cond_g > tol_cond):
        raise ValueError('Reduced matrices are ill-conditioned/singular beyond tolerance.')
    eigvals, eigvecs = scipy.linalg.eig(K_e_ff, -K_g_ff, right=True)
    complex_tol = 1e-08
    finite_mask = np.isfinite(eigvals)
    real_part = eigvals.real
    imag_part = np.abs(eigvals.imag)
    rel_scale = np.maximum(1.0, np.abs(real_part))
    nearly_real_mask = imag_part <= complex_tol * rel_scale
    positive_mask = real_part > 0.0
    mask = finite_mask & nearly_real_mask & positive_mask
    idx_candidates = np.flatnonzero(mask)
    if idx_candidates.size == 0:
        raise ValueError('No positive eigenvalue is found.')
    candidate_values = real_part[idx_candidates]
    i_local = int(np.argmin(candidate_values))
    i_sel = int(idx_candidates[i_local])
    lam = eigvals[i_sel]
    phi_ff = eigvecs[:, i_sel]
    if np.abs(lam.imag) > complex_tol * max(1.0, abs(lam.real)):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    vec_real_scale = np.maximum(1.0, np.max(np.abs(phi_ff.real)) if phi_ff.real.size else 1.0)
    if np.max(np.abs(phi_ff.imag)) > complex_tol * vec_real_scale:
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    deformed_shape_vector = np.zeros(ndof, dtype=float)
    deformed_shape_vector[free_idx] = phi_ff.real
    return (float(lam.real), deformed_shape_vector)