def eigenvalue_analysis_msa_3D(K_e_global, K_g_global, boundary_conditions, n_nodes):
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    K_e_ff = K_e_global[np.ix_(free, free)]
    K_g_ff = K_g_global[np.ix_(free, free)]
    if np.linalg.cond(K_e_ff) > 1e+16 or np.linalg.cond(K_g_ff) > 1e+16:
        raise ValueError('Reduced stiffness matrices are ill-conditioned or singular')
    (eigenvals, eigenvecs) = scipy.linalg.eig(-K_e_ff, K_g_ff)
    if np.any(np.abs(np.imag(eigenvals)) > 1e-10):
        raise ValueError('Eigenvalues contain non-negligible complex parts')
    if np.any(np.abs(np.imag(eigenvecs)) > 1e-10):
        raise ValueError('Eigenvectors contain non-negligible complex parts')
    real_eigenvals = np.real(eigenvals)
    positive_mask = real_eigenvals > 0
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalues found')
    min_positive_idx = np.argmin(real_eigenvals[positive_mask])
    elastic_critical_load_factor = real_eigenvals[positive_mask][min_positive_idx]
    critical_eigenvec = np.real(eigenvecs[:, positive_mask][:, min_positive_idx])
    deformed_shape_vector = np.zeros(6 * n_nodes)
    deformed_shape_vector[free] = critical_eigenvec
    return (elastic_critical_load_factor, deformed_shape_vector)