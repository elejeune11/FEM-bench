def MSA_3D_solve_eigenvalue_CC1_H1_T3(K_e_global: np.ndarray, K_g_global: np.ndarray, boundary_conditions: dict, n_nodes: int):
    full_mask = np.ones(6 * n_nodes, dtype=bool)
    for (node_idx, bc) in boundary_conditions.items():
        start_idx = node_idx * 6
        end_idx = start_idx + 6
        full_mask[start_idx:end_idx] = ~np.array(bc, dtype=bool)
    free_dofs = np.where(full_mask)[0]
    n_free = len(free_dofs)
    K_e_ff = K_e_global[np.ix_(free_dofs, free_dofs)]
    K_g_ff = K_g_global[np.ix_(free_dofs, free_dofs)]
    cond_K_e = np.linalg.cond(K_e_ff)
    cond_K_g = np.linalg.cond(K_g_ff)
    if cond_K_e > 1e+16 or cond_K_g > 1e+16:
        raise ValueError('Reduced matrices are ill-conditioned/singular beyond tolerance')
    try:
        (eigenvals, eigenvecs) = scipy.linalg.eig(K_e_ff, -K_g_ff)
    except:
        raise ValueError('Eigenvalue problem could not be solved')
    if np.any(np.abs(np.imag(eigenvals)) > 1e-12) or np.any(np.abs(np.imag(eigenvecs)) > 1e-12):
        raise ValueError('Eigenpairs contain non-negligible complex parts')
    eigenvals = np.real(eigenvals)
    eigenvecs = np.real(eigenvecs)
    positive_eigenvals = eigenvals[eigenvals > 0]
    if len(positive_eigenvals) == 0:
        raise ValueError('No positive eigenvalue found')
    min_positive_idx = np.argmin(positive_eigenvals)
    elastic_critical_load_factor = positive_eigenvals[min_positive_idx]
    original_idx = np.where(eigenvals == elastic_critical_load_factor)[0]
    if len(original_idx) == 0:
        original_idx = np.argmin(np.abs(eigenvals - elastic_critical_load_factor))
    else:
        original_idx = original_idx[0]
    mode_free = eigenvecs[:, original_idx]
    deformed_shape_vector = np.zeros(6 * n_nodes)
    deformed_shape_vector[free_dofs] = mode_free
    return (elastic_critical_load_factor, deformed_shape_vector)