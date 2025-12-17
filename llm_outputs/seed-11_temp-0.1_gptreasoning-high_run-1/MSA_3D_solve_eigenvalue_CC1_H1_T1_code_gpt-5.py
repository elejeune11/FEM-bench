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
    n_dof = 6 * n_nodes
    K_e_global = np.asarray(K_e_global)
    K_g_global = np.asarray(K_g_global)
    if K_e_global.shape != (n_dof, n_dof) or K_g_global.shape != (n_dof, n_dof):
        raise ValueError('Input matrices must be square with shape (6*n_nodes, 6*n_nodes).')
    _, free = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    if free.size == 0:
        raise ValueError('No free DOFs available to solve the eigenvalue problem.')
    K_e_ff = K_e_global[np.ix_(free, free)]
    K_g_ff = K_g_global[np.ix_(free, free)]

    def _cond_2(mat: np.ndarray) -> float:
        s = np.linalg.svd(mat, compute_uv=False)
        if s.size == 0:
            return np.inf
        s_min = s[-1]
        if s_min == 0 or not np.isfinite(s_min):
            return np.inf
        s_max = s[0]
        return float(s_max / s_min)
    cond_tol = 1e+16
    cond_e = _cond_2(K_e_ff)
    cond_g = _cond_2(K_g_ff)
    if not np.isfinite(cond_e) or cond_e > cond_tol or (not np.isfinite(cond_g)) or (cond_g > cond_tol):
        raise ValueError('Reduced matrices are ill-conditioned or singular beyond tolerance 1e16.')
    evals, evecs = scipy.linalg.eig(K_e_ff, -K_g_ff)
    imag_tol = 1e-08
    real_mask = np.abs(evals.imag) <= imag_tol * np.maximum(1.0, np.abs(evals.real))
    positive_mask = (evals.real > 0) & real_mask
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalue found.')
    positive_evals = evals.real[positive_mask]
    pos_indices = np.where(positive_mask)[0]
    idx_rel = int(np.argmin(positive_evals))
    idx = int(pos_indices[idx_rel])
    lambda_min = float(evals[idx].real)
    phi_free = evecs[:, idx]
    if np.linalg.norm(phi_free.imag) > imag_tol * max(1.0, np.linalg.norm(phi_free.real)):
        raise ValueError('Selected eigenpair has non-negligible complex parts.')
    deformed_shape_vector = np.zeros(n_dof, dtype=float)
    deformed_shape_vector[free] = phi_free.real
    return (lambda_min, deformed_shape_vector)