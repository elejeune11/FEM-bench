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
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    if free.size == 0:
        raise ValueError('No free DOFs after applying boundary conditions.')
    Ke_ff = np.asarray(K_e_global, dtype=float)[np.ix_(free, free)]
    Kg_ff = np.asarray(K_g_global, dtype=float)[np.ix_(free, free)]
    if not (np.all(np.isfinite(Ke_ff)) and np.all(np.isfinite(Kg_ff))):
        raise ValueError('Reduced matrices contain non-finite values.')
    tol_cond = 1e+16
    try:
        cond_Ke = np.linalg.cond(Ke_ff)
    except Exception:
        cond_Ke = np.inf
    try:
        cond_Kg = np.linalg.cond(Kg_ff)
    except Exception:
        cond_Kg = np.inf
    if not np.isfinite(cond_Ke) or not np.isfinite(cond_Kg) or cond_Ke > tol_cond or (cond_Kg > tol_cond):
        raise ValueError('Reduced matrices are ill-conditioned or singular beyond tolerance.')
    try:
        (eigvals, eigvecs) = scipy.linalg.eig(Ke_ff, -Kg_ff)
    except Exception as e:
        raise ValueError('Failed to solve the generalized eigenproblem.') from e
    eigvals = np.asarray(eigvals)
    complex_tol = 1e-08
    if np.any(np.abs(eigvals.imag) > complex_tol * (1.0 + np.abs(eigvals.real))):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    real_vals = eigvals.real
    finite_mask = np.isfinite(real_vals)
    pos_mask = (real_vals > 0.0) & finite_mask
    if not np.any(pos_mask):
        raise ValueError('No positive eigenvalue found.')
    candidate_indices = np.where(pos_mask)[0]
    min_idx_local = np.argmin(real_vals[candidate_indices])
    min_idx = candidate_indices[min_idx_local]
    lambda_min = float(real_vals[min_idx])
    phi_free = eigvecs[:, min_idx]
    if np.max(np.abs(phi_free.imag)) > complex_tol * (1.0 + np.max(np.abs(phi_free.real))):
        raise ValueError('Eigenpairs contain non-negligible complex parts in eigenvectors.')
    phi_free = np.real(phi_free)
    deformed_shape_vector = np.zeros(6 * n_nodes, dtype=float)
    deformed_shape_vector[free] = phi_free
    return (lambda_min, deformed_shape_vector)