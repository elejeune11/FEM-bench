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
    n_dof = 6 * int(n_nodes)
    if K_e_global.shape != (n_dof, n_dof):
        raise ValueError('K_e_global has incorrect shape for provided n_nodes.')
    if K_g_global.shape != (n_dof, n_dof):
        raise ValueError('K_g_global has incorrect shape for provided n_nodes.')
    constrained = np.zeros(n_dof, dtype=bool)
    if boundary_conditions is None:
        boundary_conditions = {}
    for (node, bc) in boundary_conditions.items():
        if not 0 <= int(node) < n_nodes:
            raise ValueError('boundary_conditions contains invalid node index.')
        arr = np.asarray(bc, dtype=bool)
        if arr.size != 6:
            raise ValueError('Each boundary condition entry must be length 6.')
        start = int(node) * 6
        constrained[start:start + 6] = arr
    free_idx = np.where(~constrained)[0]
    if free_idx.size == 0:
        raise ValueError('No free DOFs available to solve eigenproblem.')
    K_e_ff = np.asarray(K_e_global, dtype=float)[np.ix_(free_idx, free_idx)]
    K_g_ff = np.asarray(K_g_global, dtype=float)[np.ix_(free_idx, free_idx)]
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
        raise ValueError('Reduced matrices are ill-conditioned or singular beyond tolerance.')
    try:
        (eigvals, eigvecs) = scipy.linalg.eig(K_e_ff, -K_g_ff)
    except Exception as e:
        raise ValueError(f'Eigenvalue solver failed: {e}')
    if np.any(~np.isfinite(eigvals)):
        raise ValueError('Eigenvalue computation produced non-finite values.')
    tol_complex = 1e-08
    eigvals_real = np.real(eigvals)
    eigvals_imag = np.imag(eigvals)
    max_real_ev = np.maximum(1.0, np.abs(eigvals_real))
    if np.any(np.abs(eigvals_imag) > tol_complex * max_real_ev):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    eigvecs_imag = np.imag(eigvecs)
    eigvecs_real = np.real(eigvecs)
    for i in range(eigvecs.shape[1]):
        col_real_max = np.max(np.abs(eigvecs_real[:, i]))
        col_imag_max = np.max(np.abs(eigvecs_imag[:, i]))
        if col_imag_max > tol_complex * max(1.0, col_real_max):
            raise ValueError('Eigenpairs contain non-negligible complex parts.')
    eigvals = eigvals_real
    eigvecs = eigvecs_real
    tol_positive = 1e-12
    positive_indices = np.where(eigvals > tol_positive)[0]
    if positive_indices.size == 0:
        raise ValueError('No positive eigenvalue found.')
    positive_values = eigvals[positive_indices]
    min_pos_idx_local = int(np.argmin(positive_values))
    chosen_idx = positive_indices[min_pos_idx_local]
    elastic_critical_load_factor = float(eigvals[chosen_idx])
    phi_reduced = eigvecs[:, chosen_idx].astype(float)
    deformed_shape_vector = np.zeros(n_dof, dtype=float)
    deformed_shape_vector[free_idx] = phi_reduced
    return (elastic_critical_load_factor, deformed_shape_vector)