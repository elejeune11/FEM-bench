def MSA_3D_solve_eigenvalue_CC1_H1_T1(K_e_global: np.ndarray, K_g_global: np.ndarray, boundary_conditions: dict, n_nodes: int):
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
    boundary_conditions : dict[int, array-like of bool]
        Dictionary mapping each node index (0-based) to a 6-element boolean array
        defining the constrained DOFs:
            True  → fixed (prescribed zero displacement/rotation)
            False → free (unknown)
        Nodes not listed are assumed fully free.
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
    Helper Functions
    ----------------
        Identifies which global DOFs are fixed and which are free, returning
        sorted integer index arrays (`fixed`, `free`). This helper ensures
        consistency between the nodal boundary-condition specification and the
        DOF layout assumed here.
    Raises
    ------
    ValueError
            Use a tolerence of 1e16
    """
    import numpy as np
    import scipy
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    n_dof = 6 * n_nodes
    if K_e_global.shape != (n_dof, n_dof) or K_g_global.shape != (n_dof, n_dof):
        raise ValueError('Global stiffness matrix shapes do not match number of nodes.')
    free = np.asarray(free, dtype=int)
    if free.size == 0:
        raise ValueError('No free degrees of freedom to solve eigenproblem.')
    K_e_ff = K_e_global[np.ix_(free, free)]
    K_g_ff = K_g_global[np.ix_(free, free)]
    try:
        cond_e = np.linalg.cond(K_e_ff)
    except Exception:
        cond_e = np.inf
    try:
        cond_g = np.linalg.cond(K_g_ff)
    except Exception:
        cond_g = np.inf
    tol_cond = 1e+16
    if cond_e > tol_cond or cond_g > tol_cond or (not np.isfinite(cond_e)) or (not np.isfinite(cond_g)):
        raise ValueError('Reduced matrices are ill-conditioned/singular beyond tolerance.')
    (w, v) = scipy.linalg.eig(K_e_ff, K_g_ff)
    eps_complex = 1e-08
    if w.size == 0:
        raise ValueError('Eigenvalue solver returned no eigenvalues.')
    imag_w = np.abs(np.imag(w))
    real_w = np.abs(np.real(w))
    thresh_w = eps_complex * np.maximum(1.0, real_w)
    if np.any(imag_w > thresh_w):
        raise ValueError('Eigenpairs contain non-negligible complex parts (eigenvalues).')
    imag_v = np.abs(np.imag(v))
    real_v = np.abs(np.real(v))
    max_real_v = np.maximum(1.0, np.max(real_v) if real_v.size > 0 else 1.0)
    if np.any(imag_v > eps_complex * max_real_v):
        raise ValueError('Eigenpairs contain non-negligible complex parts (eigenvectors).')
    w_real = np.real(w)
    lambdas = -w_real
    pos_tol = 1e-12
    pos_indices = np.where(lambdas > pos_tol)[0]
    if pos_indices.size == 0:
        raise ValueError('No positive eigenvalue found.')
    lambdas_pos = lambdas[pos_indices]
    idx_in_pos = int(np.argmin(lambdas_pos))
    chosen_idx = int(pos_indices[idx_in_pos])
    elastic_critical_load_factor = float(lambdas[chosen_idx])
    mode_free = np.real(v[:, chosen_idx])
    deformed_shape_vector = np.zeros((n_dof,), dtype=float)
    deformed_shape_vector[free] = mode_free
    return (elastic_critical_load_factor, deformed_shape_vector)