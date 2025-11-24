def eigenvalue_analysis_msa_3D(K_e_global, K_g_global, boundary_conditions, n_nodes):
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    K_e_ff = K_e_global[np.ix_(free, free)]
    K_g_ff = K_g_global[np.ix_(free, free)]
    if np.linalg.cond(K_e_ff) > 1e+16 or np.linalg.cond(K_g_ff) > 1e+16:
        raise ValueError('Reduced matrices are ill-conditioned/singular beyond tolerance.')
    (eigenvalues, eigenvectors) = scipy.linalg.eig(K_e_ff, -K_g_ff)
    if np.any(np.abs(np.imag(eigenvalues)) > 1e-10):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    positive_eigenvalues = eigenvalues[np.real(eigenvalues) > 0]
    if positive_eigenvalues.size == 0:
        raise ValueError('No positive eigenvalue found.')
    min_positive_eigenvalue = np.min(np.real(positive_eigenvalues))
    min_index = np.where(np.real(eigenvalues) == min_positive_eigenvalue)[0][0]
    deformed_shape_vector_free = np.real(eigenvectors[:, min_index])
    deformed_shape_vector = np.zeros(6 * n_nodes)
    deformed_shape_vector[free] = deformed_shape_vector_free
    return (min_positive_eigenvalue, deformed_shape_vector)