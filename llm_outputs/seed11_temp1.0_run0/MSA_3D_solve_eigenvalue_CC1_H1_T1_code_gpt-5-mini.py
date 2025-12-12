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
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    n_dof = 6 * n_nodes
    if K_e_global.shape != (n_dof, n_dof) or K_g_global.shape != (n_dof, n_dof):
        raise ValueError('Input global stiffness matrices must have shape (6*n_nodes, 6*n_nodes).')
    if free.size == 0:
        raise ValueError('No free degrees of freedom to solve eigenproblem.')
    K_e_ff = K_e_global[np.ix_(free, free)]
    K_g_ff = K_g_global[np.ix_(free, free)]
    cond_tol = 1e+16
    try:
        cond_e = np.linalg.cond(K_e_ff)
    except Exception:
        cond_e = np.inf
    try:
        cond_g = np.linalg.cond(K_g_ff)
    except Exception:
        cond_g = np.inf
    if not np.isfinite(cond_e) or cond_e > cond_tol or (not np.isfinite(cond_g)) or (cond_g > cond_tol):
        raise ValueError('Reduced matrices are ill-conditioned/singular beyond tolerance.')
    try:
        (w, v) = scipy.linalg.eig(K_e_ff, K_g_ff)
    except Exception as e:
        raise ValueError('Eigenvalue solution failed: ' + str(e))
    w = np.asarray(w)
    v = np.asarray(v)
    imag_tol = 1e-08
    if np.any(np.abs(np.imag(w)) > imag_tol) or np.any(np.abs(np.imag(v)) > imag_tol):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    w = np.real(w)
    v = np.real(v)
    lambdas = -w
    finite_mask = np.isfinite(lambdas)
    positive_mask = finite_mask & (lambdas > 1e-12)
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalue found.')
    pos_indices = np.where(positive_mask)[0]
    chosen_index = pos_indices[np.argmin(lambdas[pos_indices])]
    elastic_critical_load_factor = float(lambdas[chosen_index])
    deformed_shape_vector = np.zeros(n_dof, dtype=float)
    phi_ff = v[:, chosen_index]
    deformed_shape_vector[free] = phi_ff
    return (elastic_critical_load_factor, deformed_shape_vector)