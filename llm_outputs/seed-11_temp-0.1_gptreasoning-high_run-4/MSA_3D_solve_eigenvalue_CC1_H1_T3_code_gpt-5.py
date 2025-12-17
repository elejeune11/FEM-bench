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
    import numpy as np
    import scipy
    total_dofs = 6 * n_nodes
    K_e = np.asarray(K_e_global)
    K_g = np.asarray(K_g_global)
    fixed_mask = np.zeros(total_dofs, dtype=bool)
    if boundary_conditions is not None:
        for node_idx, dof_flags in boundary_conditions.items():
            if not 0 <= node_idx < n_nodes:
                continue
            flags = np.asarray(dof_flags, dtype=bool).reshape(-1)
            if flags.size < 6:
                pad = np.zeros(6, dtype=bool)
                pad[:flags.size] = flags
                flags = pad
            elif flags.size > 6:
                flags = flags[:6]
            base = 6 * node_idx
            fixed_mask[base:base + 6] = flags
    free_idx = np.nonzero(~fixed_mask)[0]
    if free_idx.size == 0:
        raise ValueError('No positive eigenvalue is found.')
    K_e_ff = K_e[np.ix_(free_idx, free_idx)]
    K_g_ff = K_g[np.ix_(free_idx, free_idx)]
    tol_cond = 1e+16
    try:
        cond_e = np.linalg.cond(K_e_ff)
        cond_g = np.linalg.cond(K_g_ff)
    except Exception:
        raise ValueError('The reduced matrices are ill-conditioned/singular beyond tolerance.')
    if not (np.isfinite(cond_e) and np.isfinite(cond_g)):
        raise ValueError('The reduced matrices are ill-conditioned/singular beyond tolerance.')
    if cond_e > tol_cond or cond_g > tol_cond:
        raise ValueError('The reduced matrices are ill-conditioned/singular beyond tolerance.')
    A = -K_e_ff
    B = K_g_ff
    w, vecs = scipy.linalg.eig(A, B)
    complex_tol = 1e-08
    w_real = w.real
    w_imag = w.imag
    finite_mask = np.isfinite(w_real) & np.isfinite(w_imag)
    imag_small_mask = np.abs(w_imag) <= complex_tol * np.maximum(1.0, np.abs(w_real))
    pos_mask = w_real > 0.0
    candidates = np.where(finite_mask & imag_small_mask & pos_mask)[0]
    if candidates.size == 0:
        raise ValueError('No positive eigenvalue found.')
    selected_local = np.argmin(w_real[candidates])
    idx = candidates[selected_local]
    eigval = w[idx]
    if np.abs(eigval.imag) > complex_tol * max(1.0, np.abs(eigval.real)):
        raise ValueError('Eigenpair contains non-negligible complex parts.')
    mode_vec = vecs[:, idx]
    v_imag_norm = np.linalg.norm(mode_vec.imag)
    v_real_norm = np.linalg.norm(mode_vec.real)
    if v_imag_norm > complex_tol * max(1.0, v_real_norm):
        raise ValueError('Eigenpair contains non-negligible complex parts.')
    deformed_shape_vector = np.zeros(total_dofs, dtype=float)
    deformed_shape_vector[free_idx] = mode_vec.real
    return (float(eigval.real), deformed_shape_vector)