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
    (fixed_dofs, free_dofs) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    K_e_ff = K_e_global[np.ix_(free_dofs, free_dofs)]
    K_g_ff = K_g_global[np.ix_(free_dofs, free_dofs)]
    if np.linalg.cond(K_e_ff) > 1e+16 or np.linalg.cond(K_g_ff) > 1e+16:
        raise ValueError('Reduced matrices are ill-conditioned or singular')
    (eigenvals, eigenvecs) = scipy.linalg.eig(-K_e_ff, K_g_ff)
    if np.any(np.abs(np.imag(eigenvals)) > 1e-10):
        raise ValueError('Eigenvalues contain non-negligible complex parts')
    eigenvals = np.real(eigenvals)
    positive_mask = eigenvals > 0
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalues found')
    min_positive_idx = np.argmin(eigenvals[positive_mask])
    elastic_critical_load_factor = eigenvals[positive_mask][min_positive_idx]
    critical_mode = np.real(eigenvecs[:, positive_mask][:, min_positive_idx])
    global_mode = np.zeros(6 * n_nodes)
    global_mode[free_dofs] = critical_mode
    return (elastic_critical_load_factor, global_mode)