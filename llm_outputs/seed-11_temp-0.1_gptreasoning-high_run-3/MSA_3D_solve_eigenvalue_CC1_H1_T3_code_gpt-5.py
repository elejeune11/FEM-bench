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
    ndof = 6 * n_nodes
    fixed_mask = np.zeros(ndof, dtype=bool)
    if boundary_conditions is not None:
        for node, bc in boundary_conditions.items():
            if node < 0 or node >= n_nodes:
                continue
            bc_arr = np.array(bc, dtype=bool).ravel()
            bc_fixed = np.zeros(6, dtype=bool)
            m = min(6, bc_arr.size)
            if m > 0:
                bc_fixed[:m] = bc_arr[:m]
            idx = 6 * node + np.arange(6)
            fixed_mask[idx] = bc_fixed
    free_mask = ~fixed_mask
    free_idx = np.flatnonzero(free_mask)
    if free_idx.size == 0:
        raise ValueError('No positive eigenvalue found.')
    K_e_ff = K_e_global[np.ix_(free_idx, free_idx)]
    K_g_ff = K_g_global[np.ix_(free_idx, free_idx)]

    def _cond2(M: np.ndarray) -> float:
        if M.size == 0:
            return np.inf
        s = np.linalg.svd(M, compute_uv=False)
        if s.size == 0 or s[-1] == 0:
            return np.inf
        return float(s[0] / s[-1])
    tol_cond = 1e+16
    cond_Ke = _cond2(K_e_ff)
    if not np.isfinite(cond_Ke) or cond_Ke > tol_cond:
        raise ValueError('Reduced matrices are ill-conditioned or singular beyond tolerance (1e16).')
    try:
        eigvals, eigvecs = scipy.linalg.eig(-K_e_ff, K_g_ff)
    except Exception as e:
        raise ValueError('Failed to solve the generalized eigenproblem.') from e
    if eigvals.size == 0:
        raise ValueError('No positive eigenvalue found.')
    finite_mask = np.isfinite(eigvals)
    eigvals = eigvals[finite_mask]
    eigvecs = eigvecs[:, finite_mask]
    if eigvals.size == 0:
        raise ValueError('No positive eigenvalue found.')
    real_part = eigvals.real
    imag_part = eigvals.imag
    im_tol = 1e-08
    small_imag_mask = np.abs(imag_part) <= im_tol * np.maximum(1.0, np.abs(real_part))
    pos_mask = real_part > 1e-12
    valid_mask = small_imag_mask & pos_mask
    if not np.any(valid_mask):
        raise ValueError('No positive eigenvalue found.')
    valid_real = real_part[valid_mask]
    min_idx_local = int(np.argmin(valid_real))
    selected_global_idx = np.flatnonzero(valid_mask)[min_idx_local]
    lambda_min = float(real_part[selected_global_idx])
    phi_free = eigvecs[:, selected_global_idx]
    max_mag = float(np.max(np.abs(phi_free))) if phi_free.size > 0 else 0.0
    if np.max(np.abs(np.imag(phi_free))) > im_tol * max(1.0, max_mag):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    phi_free = np.real(phi_free).astype(float, copy=False)
    deformed_shape_vector = np.zeros(ndof, dtype=float)
    deformed_shape_vector[free_idx] = phi_free
    return (lambda_min, deformed_shape_vector)