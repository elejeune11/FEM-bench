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
    total_dofs = 6 * n_nodes
    is_constrained = np.zeros(total_dofs, dtype=bool)
    for (node_idx, constraints) in boundary_conditions.items():
        start_idx = 6 * node_idx
        is_constrained[start_idx:start_idx + 6] = constraints
    free_dofs = ~is_constrained
    K_e_ff = K_e_global[np.ix_(free_dofs, free_dofs)]
    K_g_ff = K_g_global[np.ix_(free_dofs, free_dofs)]
    cond_K_e_ff = np.linalg.cond(K_e_ff)
    cond_K_g_ff = np.linalg.cond(K_g_ff)
    if cond_K_e_ff > 1e+16 or cond_K_g_ff > 1e+16:
        raise ValueError('Reduced matrices are ill-conditioned/singular beyond tolerance.')
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eig(K_e_ff, -K_g_ff)
    except np.linalg.LinAlgError:
        raise ValueError('Eigenproblem could not be solved due to singular matrices.')
    eigenvalues = np.real_if_close(eigenvalues, tol=1000000.0)
    if not np.all(np.isreal(eigenvalues)):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    eigenvalues = eigenvalues[np.isreal(eigenvalues)]
    valid_indices = eigenvalues > 0
    if not np.any(valid_indices):
        raise ValueError('No positive eigenvalue found.')
    min_index = np.argmin(eigenvalues[valid_indices])
    selected_index = np.where(valid_indices)[0][min_index]
    lambda_min = eigenvalues[selected_index]
    phi_ff = eigenvectors[:, selected_index]
    deformed_shape_vector = np.zeros(total_dofs)
    deformed_shape_vector[free_dofs] = phi_ff
    return (lambda_min, deformed_shape_vector)