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
    import numpy as np
    import scipy
    total_dof = 6 * n_nodes
    K_e = np.asarray(K_e_global)
    K_g = np.asarray(K_g_global)
    if K_e.shape != (total_dof, total_dof) or K_g.shape != (total_dof, total_dof):
        raise ValueError('Input matrices must have shape (6*n_nodes, 6*n_nodes).')
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    if free.size == 0:
        raise ValueError('No free degrees of freedom after applying boundary conditions.')
    K_e_ff = K_e[np.ix_(free, free)]
    K_g_ff = K_g[np.ix_(free, free)]
    cond_tol = 1e+16
    try:
        cond_Ke = np.linalg.cond(K_e_ff)
    except np.linalg.LinAlgError:
        cond_Ke = np.inf
    try:
        cond_Kg = np.linalg.cond(K_g_ff)
    except np.linalg.LinAlgError:
        cond_Kg = np.inf
    if not np.isfinite(cond_Ke) or cond_Ke > cond_tol:
        raise ValueError('Reduced elastic stiffness matrix is ill-conditioned or singular.')
    if not np.isfinite(cond_Kg) or cond_Kg > cond_tol:
        raise ValueError('Reduced geometric stiffness matrix is ill-conditioned or singular.')
    A = K_e_ff
    B = -K_g_ff
    try:
        (eigvals, eigvecs) = scipy.linalg.eig(A, B)
    except Exception as e:
        raise ValueError(f'Generalized eigenvalue solve failed: {e}')
    finite_mask = np.isfinite(eigvals)
    eigvals = eigvals[finite_mask]
    eigvecs = eigvecs[:, finite_mask]
    if eigvals.size == 0:
        raise ValueError('No finite eigenvalues found.')
    eigvals_real = eigvals.real
    eigvals_imag = eigvals.imag
    imag_tol = 1e-08
    real_mask = np.abs(eigvals_imag) <= imag_tol * (np.abs(eigvals_real) + 1.0)
    pos_mask = eigvals_real > 0.0
    valid_mask = real_mask & pos_mask
    if not np.any(valid_mask):
        raise ValueError('No positive eigenvalue is found.')
    valid_eigvals_real = eigvals_real[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    min_idx_local = np.argmin(valid_eigvals_real)
    chosen_global_index = valid_indices[min_idx_local]
    lambda_min = eigvals_real[chosen_global_index]
    phi = eigvecs[:, chosen_global_index]
    if not abs(eigvals_imag[chosen_global_index]) <= imag_tol * (abs(lambda_min) + 1.0):
        raise ValueError('Selected eigenvalue has non-negligible imaginary part.')
    if not np.all(np.abs(phi.imag) <= imag_tol * (np.abs(phi.real) + 1.0)):
        raise ValueError('Selected eigenvector has non-negligible imaginary parts.')
    phi_real = phi.real
    deformed_shape_vector = np.zeros(total_dof, dtype=float)
    deformed_shape_vector[free] = phi_real
    return (float(lambda_min), deformed_shape_vector)