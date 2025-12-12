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
    if not isinstance(n_nodes, int) or n_nodes <= 0:
        raise ValueError('n_nodes must be a positive integer.')
    n_dof = 6 * n_nodes
    K_e = np.asarray(K_e_global)
    K_g = np.asarray(K_g_global)
    if K_e.shape != (n_dof, n_dof) or K_g.shape != (n_dof, n_dof):
        raise ValueError('Input matrices must be square with shape (6*n_nodes, 6*n_nodes).')
    (_, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    if free.size == 0:
        raise ValueError('No free DOFs: system is fully constrained.')
    K_e_ff = K_e[np.ix_(free, free)]
    K_g_ff = K_g[np.ix_(free, free)]
    cond_tol = 1e+16
    try:
        cond_e = np.linalg.cond(K_e_ff)
        cond_g = np.linalg.cond(K_g_ff)
    except np.linalg.LinAlgError as err:
        raise ValueError(f'Condition number calculation failed: {err}')
    if not np.isfinite(cond_e) or cond_e > cond_tol:
        raise ValueError('Reduced elastic stiffness matrix is ill-conditioned/singular beyond tolerance.')
    if not np.isfinite(cond_g) or cond_g > cond_tol:
        raise ValueError('Reduced geometric stiffness matrix is ill-conditioned/singular beyond tolerance.')
    try:
        (eigvals, eigvecs) = scipy.linalg.eig(K_e_ff, K_g_ff)
    except Exception as err:
        raise ValueError(f'Generalized eigenvalue problem failed: {err}')
    lambdas = -eigvals
    imag_tol_scale = 1e-08
    pos_tol = 1e-12
    real_parts = lambdas.real
    imag_parts = lambdas.imag
    imag_ok = np.abs(imag_parts) <= imag_tol_scale * np.maximum(1.0, np.abs(real_parts))
    positive = real_parts > pos_tol
    mask = imag_ok & positive
    if not np.any(mask):
        raise ValueError('No positive eigenvalue found.')
    candidate_indices = np.where(mask)[0]
    min_idx_local = np.argmin(real_parts[candidate_indices])
    idx = candidate_indices[min_idx_local]
    lambda_sel = lambdas[idx]
    phi_sel = eigvecs[:, idx]
    if not np.abs(lambda_sel.imag) <= imag_tol_scale * max(1.0, abs(lambda_sel.real)):
        raise ValueError('Selected eigenvalue has non-negligible complex part.')
    max_real = max(1.0, np.max(np.abs(phi_sel.real))) if phi_sel.size > 0 else 1.0
    if np.max(np.abs(phi_sel.imag)) > imag_tol_scale * max_real:
        raise ValueError('Selected eigenvector has non-negligible complex part.')
    mode_global = np.zeros(n_dof, dtype=float)
    mode_global[free] = phi_sel.real
    return (float(lambda_sel.real), mode_global)