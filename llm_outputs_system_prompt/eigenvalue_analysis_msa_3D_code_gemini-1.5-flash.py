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
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    K_e_ff = K_e_global[np.ix_(free, free)]
    K_g_ff = K_g_global[np.ix_(free, free)]
    try:
        (w, v) = scipy.linalg.eig(K_e_ff, -K_g_ff)
    except scipy.linalg.LinAlgError:
        raise ValueError('Reduced matrices are ill-conditioned/singular.')
    w = w.real
    v = v.real
    positive_eigenvalues = w[w > 0]
    if positive_eigenvalues.size == 0:
        raise ValueError('No positive eigenvalue found.')
    min_positive_eigenvalue_index = np.argmin(positive_eigenvalues)
    elastic_critical_load_factor = positive_eigenvalues[min_positive_eigenvalue_index]
    deformed_shape_vector = np.zeros(6 * n_nodes)
    deformed_shape_vector[free] = v[:, min_positive_eigenvalue_index]
    if np.any(np.abs(w.imag) > 1e-08) or np.any(np.abs(v.imag) > 1e-08):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    return (elastic_critical_load_factor, deformed_shape_vector)