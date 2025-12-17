def MSA_3D_solve_eigenvalue_CC1_H1_T1(K_e_global: np.ndarray, K_g_global: np.ndarray, boundary_conditions: dict, n_nodes: int):
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
    Helper Functions
    ----------------
        Identifies which global DOFs are fixed and which are free, returning
        sorted integer index arrays (`fixed`, `free`). This helper ensures
        consistency between the nodal boundary-condition specification and the
        DOF layout assumed here.
    Raises
    ------
    ValueError
            Use a tolerence of 1e16
    """
    n_dof = 6 * int(n_nodes)
    if K_e_global.shape != (n_dof, n_dof) or K_g_global.shape != (n_dof, n_dof):
        raise ValueError('Input stiffness matrices must be square with shape (6*n_nodes, 6*n_nodes).')
    fixed, free = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    m = free.size
    if m == 0:
        raise ValueError('No free DOFs remain after applying boundary conditions.')
    K_e_ff = K_e_global[np.ix_(free, free)]
    K_g_ff = K_g_global[np.ix_(free, free)]
    cond_threshold = 1e+16
    cond_e = np.linalg.cond(K_e_ff)
    cond_g = np.linalg.cond(K_g_ff)
    if not np.isfinite(cond_e) or cond_e > cond_threshold or (not np.isfinite(cond_g)) or (cond_g > cond_threshold):
        raise ValueError(f'Reduced matrices ill-conditioned/singular: cond(K_e_ff)={cond_e}, cond(K_g_ff)={cond_g}, threshold={cond_threshold}')
    w, v = scipy.linalg.eig(K_e_ff, -K_g_ff)
    finite_mask = np.isfinite(w.real) & np.isfinite(w.imag)
    w = w[finite_mask]
    v = v[:, finite_mask]
    eig_imag_tol = 1e-08
    vec_imag_rel_tol = 1e-08
    real_and_pos_mask = (np.abs(w.imag) <= eig_imag_tol) & (w.real > 0.0)
    if not np.any(real_and_pos_mask):
        raise ValueError('No positive eigenvalue found.')
    w_pos = w.real[real_and_pos_mask]
    idx_candidates = np.flatnonzero(real_and_pos_mask)
    idx_min_local = np.argmin(w_pos)
    idx_min = idx_candidates[idx_min_local]
    lambda_min = float(w.real[idx_min])
    vec = v[:, idx_min]
    real_norm = np.linalg.norm(vec.real)
    imag_norm = np.linalg.norm(vec.imag)
    if real_norm == 0.0 and imag_norm != 0.0:
        raise ValueError('Selected eigenpair contains non-negligible complex parts.')
    rel_imag = imag_norm / max(real_norm, 1e-16)
    if rel_imag > vec_imag_rel_tol or np.abs(w.imag[idx_min]) > eig_imag_tol:
        raise ValueError('Selected eigenpair contains non-negligible complex parts.')
    deformed_shape_vector = np.zeros(n_dof, dtype=float)
    deformed_shape_vector[free] = vec.real
    return (lambda_min, deformed_shape_vector)