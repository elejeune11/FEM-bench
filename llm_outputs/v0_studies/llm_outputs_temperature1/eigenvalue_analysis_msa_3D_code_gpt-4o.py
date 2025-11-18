def eigenvalue_analysis_msa_3D(K_e_global, K_g_global, boundary_conditions, n_nodes):
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    K_e_ff = K_e_global[np.ix_(free, free)]
    K_g_ff = K_g_global[np.ix_(free, free)]
    if np.linalg.cond(K_e_ff) > 1e+16 or np.linalg.cond(K_g_ff) > 1e+16:
        raise ValueError('The reduced matrices are ill-conditioned/singular beyond tolerance.')
    (eigenvalues, eigenvectors) = scipy.linalg.eig(K_e_ff, -K_g_ff)
    if np.any(np.iscomplex(eigenvalues)):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    if positive_eigenvalues.size == 0:
        raise ValueError('No positive eigenvalue found.')
    elastic_critical_load_factor = np.min(positive_eigenvalues)
    min_index = np.argmin(eigenvalues)
    deformed_shape_vector = np.zeros(6 * n_nodes)
    deformed_shape_vector[free] = eigenvectors[:, min_index].real
    return (elastic_critical_load_factor, deformed_shape_vector)