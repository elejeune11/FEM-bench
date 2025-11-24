def eigenvalue_analysis_msa_3D(K_e_global, K_g_global, boundary_conditions, n_nodes):
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    K_e_ff = K_e_global[np.ix_(free, free)]
    K_g_ff = K_g_global[np.ix_(free, free)]
    cond_K_e = np.linalg.cond(K_e_ff)
    cond_K_g = np.linalg.cond(K_g_ff)
    if cond_K_e > 1e+16 or cond_K_g > 1e+16:
        raise ValueError('Reduced matrices are ill-conditioned')
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_e_ff, -K_g_ff)
    except np.linalg.LinAlgError:
        raise ValueError('Generalized eigenvalue problem failed to solve')
    positive_mask = eigenvalues > 0
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalue found')
    min_positive_idx = np.argmin(eigenvalues[positive_mask])
    elastic_critical_load_factor = eigenvalues[positive_mask][min_positive_idx]
    phi_ff = eigenvectors[:, positive_mask][:, min_positive_idx]
    if np.any(np.abs(np.imag(phi_ff)) > 1e-10) or np.abs(np.imag(elastic_critical_load_factor)) > 1e-10:
        raise ValueError('Eigenpairs contain non-negligible complex parts')
    elastic_critical_load_factor = np.real(elastic_critical_load_factor)
    phi_ff = np.real(phi_ff)
    deformed_shape_vector = np.zeros(6 * n_nodes)
    deformed_shape_vector[free] = phi_ff
    return (elastic_critical_load_factor, deformed_shape_vector)