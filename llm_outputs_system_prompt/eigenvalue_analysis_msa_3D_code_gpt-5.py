def eigenvalue_analysis_msa_3D(K_e_global, K_g_global, boundary_conditions, n_nodes):
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
    boundary_conditions : object
        Container consumed by `partition_degrees_of_freedom(boundary_conditions, n_nodes)`.
        Must define constrained DOFs such that the free set removes all rigid-body modes.
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
    n_dof = 6 * n_nodes
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    if free.size == 0:
        raise ValueError('No free degrees of freedom after applying boundary conditions.')
    if K_e_global.shape != (n_dof, n_dof) or K_g_global.shape != (n_dof, n_dof):
        raise ValueError('Global stiffness matrices have incompatible shapes.')
    K_e_ff = np.asarray(K_e_global)[np.ix_(free, free)]
    K_g_ff = np.asarray(K_g_global)[np.ix_(free, free)]
    try:
        cond_ke = np.linalg.cond(K_e_ff)
    except np.linalg.LinAlgError:
        cond_ke = np.inf
    try:
        cond_kg = np.linalg.cond(K_g_ff)
    except np.linalg.LinAlgError:
        cond_kg = np.inf
    tol_cond = 1e+16
    if not np.isfinite(cond_ke) or not np.isfinite(cond_kg) or cond_ke > tol_cond or (cond_kg > tol_cond):
        raise ValueError('Reduced matrices are ill-conditioned or singular beyond tolerance.')
    A = -K_e_ff
    B = K_g_ff
    (w, vecs) = scipy.linalg.eig(A, B, check_finite=False)
    if w.size == 0:
        raise ValueError('No eigenvalues computed.')
    tol_imag = 1e-08
    real_parts = w.real
    imag_parts = w.imag
    mask_real = np.abs(imag_parts) <= tol_imag * (1.0 + np.abs(real_parts))
    mask_positive = real_parts > 0.0
    mask_finite = np.isfinite(real_parts)
    valid_mask = mask_real & mask_positive & mask_finite
    if not np.any(valid_mask):
        raise ValueError('No positive real eigenvalue found.')
    valid_values = real_parts[valid_mask]
    idx_local = np.argmin(valid_values)
    valid_indices = np.nonzero(valid_mask)[0]
    idx = valid_indices[idx_local]
    if np.abs(imag_parts[idx]) > tol_imag * (1.0 + np.abs(real_parts[idx])):
        raise ValueError('Selected eigenpair has non-negligible complex parts.')
    lambda_min = float(real_parts[idx])
    mode_free = vecs[:, idx]
    if np.max(np.abs(mode_free.imag)) > tol_imag * (1.0 + np.max(np.abs(mode_free.real))):
        raise ValueError('Eigenvector contains non-negligible complex parts.')
    mode_full = np.zeros(n_dof, dtype=mode_free.real.dtype)
    mode_full[free] = mode_free.real
    return (lambda_min, mode_full)