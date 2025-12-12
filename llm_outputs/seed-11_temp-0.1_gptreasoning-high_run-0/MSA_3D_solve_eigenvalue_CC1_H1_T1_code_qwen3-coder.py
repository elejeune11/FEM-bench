def MSA_3D_solve_eigenvalue_CC1_H1_T1(K_e_global: np.ndarray, K_g_global: np.ndarray, boundary_conditions: dict, n_nodes: int):
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    K_e_ff = K_e_global[np.ix_(free, free)]
    K_g_ff = K_g_global[np.ix_(free, free)]
    if np.linalg.cond(K_e_ff) > 1e+16 or np.linalg.cond(K_g_ff) > 1e+16:
        raise ValueError('Reduced matrices are ill-conditioned/singular beyond tolerance.')
    (eigenvalues, eigenvectors) = scipy.linalg.eig(K_e_ff, -K_g_ff)
    if np.any(np.iscomplex(eigenvalues)) or np.any(np.iscomplex(eigenvectors)):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    positive_indices = np.where(eigenvalues > 0)[0]
    if len(positive_indices) == 0:
        raise ValueError('No positive eigenvalue found.')
    min_index = positive_indices[np.argmin(eigenvalues[positive_indices])]
    elastic_critical_load_factor = eigenvalues[min_index]
    deformed_shape_free = eigenvectors[:, min_index]
    deformed_shape_vector = np.zeros(6 * n_nodes)
    deformed_shape_vector[free] = deformed_shape_free
    return (elastic_critical_load_factor, deformed_shape_vector)