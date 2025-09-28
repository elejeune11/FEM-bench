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
    import pytest
    n_dof = 6 * n_nodes
    K_e_global = np.asarray(K_e_global)
    K_g_global = np.asarray(K_g_global)
    if K_e_global.shape != (n_dof, n_dof) or K_g_global.shape != (n_dof, n_dof):
        raise ValueError('Input matrices must be of shape (6*n_nodes, 6*n_nodes).')
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    if free.size == 0:
        raise ValueError('No free DOFs after applying boundary conditions.')
    K_e_ff = K_e_global[np.ix_(free, free)]
    K_g_ff = K_g_global[np.ix_(free, free)]
    tol_cond = 1e+16
    cond_e = np.linalg.cond(K_e_ff)
    cond_g = np.linalg.cond(K_g_ff)
    if not np.isfinite(cond_e) or cond_e > tol_cond:
        raise ValueError('Reduced elastic stiffness matrix is ill-conditioned/singular beyond tolerance.')
    if not np.isfinite(cond_g) or cond_g > tol_cond:
        raise ValueError('Reduced geometric stiffness matrix is ill-conditioned/singular beyond tolerance.')
    try:
        (eigvals, eigvecs) = scipy.linalg.eig(K_e_ff, -K_g_ff)
    except Exception as e:
        raise ValueError(f'Generalized eigenvalue solver failed: {e}')
    finite = np.isfinite(eigvals.real) & np.isfinite(eigvals.imag)
    eigvals = eigvals[finite]
    eigvecs = eigvecs[:, finite]
    if eigvals.size == 0:
        raise ValueError('No eigenvalues computed.')
    tol_complex = 1e-08
    real_like = np.abs(eigvals.imag) <= tol_complex * np.maximum(1.0, np.abs(eigvals.real))
    eigvals_real = eigvals.real[real_like]
    eigvecs_real = eigvecs[:, real_like]
    positive_mask = eigvals_real > 0.0
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalue is found.')
    pos_vals = eigvals_real[positive_mask]
    idx_local = int(np.argmin(pos_vals))
    lambda_min = float(pos_vals[idx_local])
    mode_vec_free = eigvecs_real[:, positive_mask][:, idx_local]
    if np.linalg.norm(mode_vec_free.imag) > tol_complex * max(1.0, np.linalg.norm(mode_vec_free.real)):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    mode_global = np.zeros(n_dof, dtype=float)
    mode_global[free] = mode_vec_free.real
    return (lambda_min, mode_global)