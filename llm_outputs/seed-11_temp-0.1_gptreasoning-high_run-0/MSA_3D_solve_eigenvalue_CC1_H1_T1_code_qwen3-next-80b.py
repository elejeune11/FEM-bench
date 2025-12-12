def MSA_3D_solve_eigenvalue_CC1_H1_T1(K_e_global: np.ndarray, K_g_global: np.ndarray, boundary_conditions: dict, n_nodes: int):
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    K_e_ff = K_e_global[np.ix_(free, free)]
    K_g_ff = K_g_global[np.ix_(free, free)]
    try:
        (eigenvals, eigenvecs) = scipy.linalg.eigh(K_e_ff, K_g_ff)
    except:
        raise ValueError('Reduced matrices are ill-conditioned/singular beyond tolerance')
    if np.any(np.iscomplex(eigenvals)) or np.any(np.iscomplex(eigenvecs)):
        raise ValueError('Eigenpairs contain non-negligible complex parts')
    positive_eigenvals = eigenvals[eigenvals > 0]
    if len(positive_eigenvals) == 0:
        raise ValueError('No positive eigenvalue is found')
    min_positive_idx = np.argmin(positive_eigenvals)
    elastic_critical_load_factor = positive_eigenvals[min_positive_idx]
    full_eigenvector_idx = np.where(eigenvals == elastic_critical_load_factor)[0][0]
    phi_ff = eigenvecs[:, full_eigenvector_idx]
    deformed_shape_vector = np.zeros(6 * n_nodes)
    deformed_shape_vector[free] = phi_ff
    if np.linalg.cond(K_e_ff) > 1e+16 or np.linalg.cond(K_g_ff) > 1e+16:
        raise ValueError('Reduced matrices are ill-conditioned/singular beyond tolerance')
    return (elastic_critical_load_factor, deformed_shape_vector)