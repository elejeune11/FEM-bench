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
            bc_array = np.asarray(boundary_conditions[node_idx], dtype=bool)
            for local_dof in range(6):
                if bc_array[local_dof]:
                    global_dof = 6 * node_idx + local_dof
                    constrained_dofs.append(global_dof)
    all_dofs = np.arange(total_dofs)
    free_dofs = np.setdiff1d(all_dofs, constrained_dofs)
    K_e_ff = K_e_global[np.ix_(free_dofs, free_dofs)]
    K_g_ff = K_g_global[np.ix_(free_dofs, free_dofs)]
    cond_e = np.linalg.cond(K_e_ff)
    cond_g = np.linalg.cond(K_g_ff)
    if cond_e > 1e+16 or cond_g > 1e+16:
        raise ValueError('Reduced matrices are ill-conditioned/singular beyond tolerance.')
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eig(K_e_ff, -K_g_ff)
    except np.linalg.LinAlgError:
        raise ValueError('Failed to solve the generalized eigenvalue problem.')
    if np.any(np.abs(eigenvalues.imag) > 1e-10):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    eigenvalues = eigenvalues.real
    positive_mask = eigenvalues > 1e-10
    positive_eigenvalues = eigenvalues[positive_mask]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalue found.')
    min_idx = np.argmin(positive_eigenvalues)
    elastic_critical_load_factor = positive_eigenvalues[min_idx]
    original_idx = np.where(positive_mask)[0][min_idx]
    mode_vector_free = eigenvectors[:, original_idx].real
    deformed_shape_vector = np.zeros(total_dofs)
    deformed_shape_vector[free_dofs] = mode_vector_free
    return (elastic_critical_load_factor, deformed_shape_vector)