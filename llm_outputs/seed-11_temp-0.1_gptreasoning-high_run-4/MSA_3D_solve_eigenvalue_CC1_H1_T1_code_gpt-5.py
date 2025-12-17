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
    K_e_global = np.asarray(K_e_global)
    K_g_global = np.asarray(K_g_global)
    n_dof = 6 * int(n_nodes)
    if K_e_global.shape != (n_dof, n_dof) or K_g_global.shape != (n_dof, n_dof):
        raise ValueError('Input stiffness matrices must be of shape (6*n_nodes, 6*n_nodes).')
    fixed, free = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    if free.size == 0:
        raise ValueError('No free DOFs remain after applying boundary conditions.')
    K_e_ff = K_e_global[np.ix_(free, free)]
    K_g_ff = K_g_global[np.ix_(free, free)]
    COND_TOL = 1e+16
    try:
        cond_Ke = np.linalg.cond(K_e_ff)
    except np.linalg.LinAlgError:
        cond_Ke = np.inf
    try:
        cond_Kg = np.linalg.cond(K_g_ff)
    except np.linalg.LinAlgError:
        cond_Kg = np.inf
    if not np.isfinite(cond_Ke) or cond_Ke > COND_TOL or (not np.isfinite(cond_Kg)) or (cond_Kg > COND_TOL):
        raise ValueError('Reduced stiffness matrices are ill-conditioned or singular (condition number exceeds 1e16).')
    try:
        eigvals, eigvecs = scipy.linalg.eig(K_e_ff, -K_g_ff, check_finite=True)
    except Exception as exc:
        raise ValueError('Generalized eigenvalue computation failed.') from exc
    if eigvals.size == 0:
        raise ValueError('No eigenvalues computed.')
    abs_vals = np.abs(eigvals)
    finite_abs = abs_vals[np.isfinite(abs_vals)]
    scale = float(np.max(finite_abs)) if finite_abs.size > 0 else 1.0
    scale = max(1.0, scale)
    IMAG_TOL = 1e-08 * scale
    is_effectively_real = np.abs(eigvals.imag) <= IMAG_TOL
    positive_real_mask = is_effectively_real & (eigvals.real > 0.0)
    if not np.any(positive_real_mask):
        if not np.any(is_effectively_real):
            raise ValueError('Eigenpairs contain non-negligible complex parts.')
        raise ValueError('No positive eigenvalue found.')
    positive_indices = np.where(positive_real_mask)[0]
    positive_reals = eigvals.real[positive_indices]
    idx_min_local = int(np.argmin(positive_reals))
    idx_min = positive_indices[idx_min_local]
    lambda_min = float(positive_reals[idx_min_local])
    mode_free = eigvecs[:, idx_min]
    mode_free_abs = np.abs(mode_free)
    mode_scale = float(np.max(mode_free_abs)) if mode_free_abs.size > 0 else 1.0
    mode_scale = max(1.0, mode_scale)
    if np.max(np.abs(mode_free.imag)) > 1e-08 * mode_scale:
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    mode_free_real = mode_free.real
    deformed_shape_vector = np.zeros(n_dof, dtype=float)
    deformed_shape_vector[free] = mode_free_real
    return (lambda_min, deformed_shape_vector)