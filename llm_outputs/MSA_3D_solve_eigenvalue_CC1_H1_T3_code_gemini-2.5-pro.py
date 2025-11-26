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
    n_dofs = 6 * n_nodes
    fixed_dofs_mask = np.zeros(n_dofs, dtype=bool)
    for (node_idx, constraints) in boundary_conditions.items():
        start_dof = node_idx * 6
        end_dof = start_dof + 6
        fixed_dofs_mask[start_dof:end_dof] = constraints
    free_dofs_mask = ~fixed_dofs_mask
    free_indices = np.where(free_dofs_mask)[0]
    if free_indices.size == 0:
        raise ValueError('The model is fully constrained; no free DOFs to solve for.')
    K_e_ff = K_e_global[np.ix_(free_indices, free_indices)]
    K_g_ff = K_g_global[np.ix_(free_indices, free_indices)]
    try:
        if np.linalg.cond(K_e_ff) > 1e+16:
            raise ValueError('The reduced matrices are ill-conditioned/singular beyond tolerance.')
    except np.linalg.LinAlgError:
        raise ValueError('The reduced matrices are ill-conditioned/singular beyond tolerance.')
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eig(K_e_ff, -K_g_ff)
    except (scipy.linalg.LinAlgError, ValueError) as e:
        raise ValueError(f'Eigenvalue problem failed to solve: {e}')
    complex_tol = 1e-09
    if not np.all(np.abs(eigenvalues.imag) < complex_tol) or not np.all(np.abs(eigenvectors.imag) < complex_tol):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    evals_for_min_find = np.copy(eigenvalues)
    positive_tol = 1e-09
    evals_for_min_find[evals_for_min_find <= positive_tol] = np.inf
    evals_for_min_find[~np.isfinite(evals_for_min_find)] = np.inf
    if np.all(np.isinf(evals_for_min_find)):
        raise ValueError('No positive eigenvalue is found.')
    min_idx = np.argmin(evals_for_min_find)
    elastic_critical_load_factor = eigenvalues[min_idx]
    phi_ff = eigenvectors[:, min_idx]
    deformed_shape_vector = np.zeros(n_dofs)
    deformed_shape_vector[free_dofs_mask] = phi_ff
    return (elastic_critical_load_factor, deformed_shape_vector)