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
    total_dof = 6 * n_nodes
    if K_e_global.shape != (total_dof, total_dof) or K_g_global.shape != (total_dof, total_dof):
        raise ValueError('Global stiffness matrices must be of shape (6*n_nodes, 6*n_nodes).')
    fixed = np.zeros(total_dof, dtype=bool)
    for (node_idx, dof_mask) in boundary_conditions.items():
        if dof_mask is None:
            continue
        if len(dof_mask) != 6:
            raise ValueError('Each boundary condition entry must be a 6-element boolean array.')
        base = int(node_idx) * 6
        for (i, val) in enumerate(dof_mask):
            if bool(val):
                fixed[base + i] = True
    free = ~fixed
    n_free = int(np.count_nonzero(free))
    if n_free == 0:
        raise ValueError('No free degrees of freedom to solve eigenproblem.')
    idx = np.where(free)[0]
    K_e_ff = K_e_global[np.ix_(idx, idx)]
    K_g_ff = K_g_global[np.ix_(idx, idx)]
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
        raise ValueError('Reduced matrices are ill-conditioned or singular beyond tolerance.')
    A = K_e_ff
    B = -K_g_ff
    try:
        (w, v) = scipy.linalg.eig(A, B)
    except Exception as e:
        raise ValueError('Eigenproblem solver failed: ' + str(e))
    complex_tol = 1e-08
    if np.max(np.abs(np.imag(w))) > complex_tol:
        raise ValueError('Eigenvalues contain non-negligible complex parts.')
    if np.max(np.abs(np.imag(v))) > complex_tol:
        raise ValueError('Eigenvectors contain non-negligible complex parts.')
    w_real = np.real(w)
    pos_tol = 1e-12
    positive_mask = w_real > pos_tol
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalue found.')
    pos_vals = w_real[positive_mask]
    pos_indices = np.where(positive_mask)[0]
    min_idx_local = pos_indices[int(np.argmin(pos_vals))]
    lambda_min = float(w_real[min_idx_local])
    phi_ff = np.real(v[:, min_idx_local])
    deformed_shape_vector = np.zeros(total_dof, dtype=float)
    deformed_shape_vector[idx] = phi_ff
    return (lambda_min, deformed_shape_vector)