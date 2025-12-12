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
    constrained_dofs = []
    for node_idx in range(n_nodes):
        if node_idx in boundary_conditions:
            constraints = boundary_conditions[node_idx]
            for dof_idx in range(6):
                if constraints[dof_idx]:
                    constrained_dofs.append(node_idx * 6 + dof_idx)
    all_dofs = set(range(total_dofs))
    free_dofs = sorted(list(all_dofs - set(constrained_dofs)))
    K_e_ff = K_e_global[np.ix_(free_dofs, free_dofs)]
    K_g_ff = K_g_global[np.ix_(free_dofs, free_dofs)]
    cond_K_e = np.linalg.cond(K_e_ff)
    cond_K_g = np.linalg.cond(K_g_ff)
    if cond_K_e > 1e+16 or cond_K_g > 1e+16:
        raise ValueError('Reduced matrices are ill-conditioned/singular beyond tolerance 1e16')
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eig(K_e_ff, -K_g_ff)
    except np.linalg.LinAlgError:
        raise ValueError('Failed to solve the generalized eigenproblem')
    if np.any(np.abs(np.imag(eigenvalues)) > 1e-10):
        raise ValueError('Eigenpairs contain non-negligible complex parts')
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    positive_mask = eigenvalues > 1e-10
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalue found')
    positive_eigenvalues = eigenvalues[positive_mask]
    min_positive_idx = np.argmin(positive_eigenvalues)
    positive_indices = np.where(positive_mask)[0]
    original_min_idx = positive_indices[min_positive_idx]
    elastic_critical_load_factor = eigenvalues[original_min_idx]
    mode_vector_reduced = eigenvectors[:, original_min_idx]
    deformed_shape_vector = np.zeros(total_dofs)
    for (i, dof) in enumerate(free_dofs):
        deformed_shape_vector[dof] = mode_vector_reduced[i]
    return (float(elastic_critical_load_factor), deformed_shape_vector)