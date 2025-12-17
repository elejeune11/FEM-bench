def MSA_3D_solve_eigenvalue_CC1_H1_T3(K_e_global: np.ndarray, K_g_global: np.ndarray, boundary_conditions: dict, n_nodes: int):
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
    Raises
    ------
    ValueError
            Use a tolerence of 1e16
    """
    dof_per_node = 6
    total_dof = dof_per_node * n_nodes
    K_e = np.asarray(K_e_global, dtype=float)
    K_g = np.asarray(K_g_global, dtype=float)
    if K_e.shape != (total_dof, total_dof) or K_g.shape != (total_dof, total_dof):
        raise ValueError('Input stiffness matrices must be of shape (6*n_nodes, 6*n_nodes).')
    constraints = np.zeros(total_dof, dtype=bool)
    if boundary_conditions is not None:
        for node, bc in boundary_conditions.items():
            if not (isinstance(node, int) and 0 <= node < n_nodes):
                raise ValueError('Boundary condition node index out of range.')
            bc_arr = np.asarray(bc, dtype=bool).reshape(-1)
            if bc_arr.size != dof_per_node:
                raise ValueError('Each boundary condition must have 6 boolean entries.')
            start = node * dof_per_node
            constraints[start:start + dof_per_node] = bc_arr
    free_mask = ~constraints
    if not np.any(free_mask):
        raise ValueError('No free DOFs after applying boundary conditions.')
    free_idx = np.nonzero(free_mask)[0]
    K_e_ff = K_e[np.ix_(free_idx, free_idx)]
    K_g_ff = K_g[np.ix_(free_idx, free_idx)]
    tol_cond = 1e+16
    try:
        cond_ke = np.linalg.cond(K_e_ff)
    except np.linalg.LinAlgError:
        cond_ke = np.inf
    try:
        cond_kg = np.linalg.cond(K_g_ff)
    except np.linalg.LinAlgError:
        cond_kg = np.inf
    if not np.isfinite(cond_ke) or cond_ke > tol_cond or (not np.isfinite(cond_kg)) or (cond_kg > tol_cond):
        raise ValueError('Reduced matrices are ill-conditioned/singular beyond tolerance.')
    A = -K_e_ff
    B = K_g_ff
    try:
        w, v = scipy.linalg.eig(A, B)
    except Exception as e:
        raise ValueError(f'Generalized eigenvalue problem failed: {e}')
    finite_mask = np.isfinite(w.real) & np.isfinite(w.imag)
    w = w[finite_mask]
    v = v[:, finite_mask]
    if w.size == 0:
        raise ValueError('No eigenvalues computed.')
    tol_imag_eig = 1e-08
    near_real_mask = np.abs(w.imag) <= tol_imag_eig * np.maximum(1.0, np.abs(w.real))
    positive_mask = (w.real > 0) & near_real_mask
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalue is found.')
    w_pos = w.real[positive_mask]
    idx_candidates = np.nonzero(positive_mask)[0]
    idx_min = idx_candidates[np.argmin(w_pos)]
    lambda_cr = float(w.real[idx_min])
    phi = v[:, idx_min]
    if np.linalg.norm(phi.imag) > tol_imag_eig * max(1.0, np.linalg.norm(phi.real)):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    phi_real = phi.real
    deformed_shape_vector = np.zeros(total_dof, dtype=float)
    deformed_shape_vector[free_idx] = phi_real
    return (lambda_cr, deformed_shape_vector)