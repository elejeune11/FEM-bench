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
    n_dof = 6 * n_nodes
    if free.size == 0:
        raise ValueError('No free DOFs after applying boundary conditions.')
    K_e_ff = np.asarray(K_e_global)[np.ix_(free, free)]
    K_g_ff = np.asarray(K_g_global)[np.ix_(free, free)]
    cond_tol = 1e+16

    def _is_well_conditioned(M):
        try:
            c = np.linalg.cond(M)
        except Exception:
            return (False, np.inf)
        if not np.isfinite(c):
            return (False, c)
        return (c <= cond_tol, c)
    (ok_e, cond_e) = _is_well_conditioned(K_e_ff)
    (ok_g, cond_g) = _is_well_conditioned(K_g_ff)
    if not ok_e or not ok_g:
        raise ValueError('Reduced matrices are ill-conditioned or singular beyond tolerance.')
    try:
        (w, v) = scipy.linalg.eig(K_e_ff, -K_g_ff)
    except Exception as e:
        raise ValueError('Failed to solve generalized eigenproblem.') from e
    im_tol = 1e-08
    real_w = w.real
    imag_w = w.imag
    real_mask = np.abs(imag_w) <= im_tol * np.maximum(1.0, np.abs(real_w))
    pos_mask = real_w > 0.0
    mask = real_mask & pos_mask
    if not np.any(mask):
        raise ValueError('No positive eigenvalue is found.')
    indices = np.where(mask)[0]
    idx_min_local = np.argmin(real_w[indices])
    idx = indices[idx_min_local]
    lam = real_w[idx]
    if np.abs(w[idx].imag) > im_tol * max(1.0, abs(lam)):
        raise ValueError('Selected eigenvalue has non-negligible complex part.')
    phi_f = v[:, idx]
    if np.linalg.norm(phi_f.imag) > im_tol * max(1.0, np.linalg.norm(phi_f.real)):
        raise ValueError('Selected eigenvector has non-negligible complex part.')
    phi_f = phi_f.real
    phi_global = np.zeros(n_dof, dtype=float)
    phi_global[free] = phi_f
    return (float(lam), phi_global)