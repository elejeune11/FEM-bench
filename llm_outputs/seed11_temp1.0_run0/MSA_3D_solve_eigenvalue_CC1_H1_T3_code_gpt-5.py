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
    dof_per_node = 6
    if not isinstance(n_nodes, int) or n_nodes <= 0:
        raise ValueError('n_nodes must be a positive integer.')
    n_dof = dof_per_node * n_nodes
    if not (isinstance(K_e_global, np.ndarray) and isinstance(K_g_global, np.ndarray)):
        raise ValueError('K_e_global and K_g_global must be numpy arrays.')
    if K_e_global.shape != (n_dof, n_dof) or K_g_global.shape != (n_dof, n_dof):
        raise ValueError('K_e_global and K_g_global must be square matrices of shape (6*n_nodes, 6*n_nodes).')
    fixed_mask = np.zeros(n_dof, dtype=bool)
    if boundary_conditions is not None:
        for (node, bc) in boundary_conditions.items():
            if not isinstance(node, int):
                raise ValueError('Boundary condition keys (node indices) must be integers.')
            if node < 0 or node >= n_nodes:
                raise ValueError(f'Boundary condition provided for invalid node index {node}.')
            bc_arr = np.asarray(bc, dtype=bool).ravel()
            if bc_arr.size != dof_per_node:
                raise ValueError(f'Boundary condition for node {node} must have 6 boolean entries.')
            start = node * dof_per_node
            fixed_mask[start:start + dof_per_node] = bc_arr
    free_mask = ~fixed_mask
    free_idx = np.nonzero(free_mask)[0]
    if free_idx.size == 0:
        raise ValueError('No free DOFs remain after applying boundary conditions.')
    K_e_ff = np.asarray(K_e_global[np.ix_(free_idx, free_idx)], dtype=float)
    K_g_ff = np.asarray(K_g_global[np.ix_(free_idx, free_idx)], dtype=float)
    if not (np.all(np.isfinite(K_e_ff)) and np.all(np.isfinite(K_g_ff))):
        raise ValueError('Reduced matrices contain non-finite values.')
    tol_cond = 1e+16
    try:
        cond_e = np.linalg.cond(K_e_ff)
    except Exception:
        cond_e = np.inf
    try:
        cond_g = np.linalg.cond(K_g_ff)
    except Exception:
        cond_g = np.inf
    if not np.isfinite(cond_e) or not np.isfinite(cond_g) or cond_e > tol_cond or (cond_g > tol_cond):
        raise ValueError('Reduced matrices are ill-conditioned or singular beyond the tolerance of 1e16.')
    try:
        (eigvals_sigma, eigvecs) = scipy.linalg.eig(K_e_ff, K_g_ff)
    except Exception as e:
        raise ValueError(f'Generalized eigenproblem failed to solve: {e}')
    lambdas = -eigvals_sigma
    imag_tol = 1e-08
    real_parts = np.real(lambdas)
    imag_parts = np.imag(lambdas)
    denom = np.maximum(1.0, np.abs(real_parts))
    is_nearly_real = np.abs(imag_parts) <= imag_tol * denom
    is_positive = real_parts > 0.0
    valid = is_nearly_real & is_positive
    if not np.any(valid):
        raise ValueError('No positive eigenvalue is found.')
    valid_indices = np.where(valid)[0]
    min_idx_local = np.argmin(real_parts[valid_indices])
    sel_idx = valid_indices[min_idx_local]
    lambda_cr = float(real_parts[sel_idx])
    phi_ff = eigvecs[:, sel_idx]
    if phi_ff.ndim != 1:
        phi_ff = phi_ff.reshape(-1)
    phi_ff_real = np.real(phi_ff)
    phi_ff_imag = np.imag(phi_ff)
    vec_scale = np.maximum(1.0, np.max(np.abs(phi_ff_real)))
    if np.max(np.abs(phi_ff_imag)) > imag_tol * vec_scale:
        raise ValueError('Selected eigenvector contains non-negligible complex parts.')
    deformed_shape_vector = np.zeros(n_dof, dtype=float)
    deformed_shape_vector[free_idx] = phi_ff_real
    return (lambda_cr, deformed_shape_vector)