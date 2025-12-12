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
    if not isinstance(n_nodes, int) or n_nodes <= 0:
        raise ValueError('n_nodes must be a positive integer.')
    n_dof = 6 * n_nodes
    K_e = np.asarray(K_e_global, dtype=float)
    K_g = np.asarray(K_g_global, dtype=float)
    if K_e.shape != (n_dof, n_dof) or K_g.shape != (n_dof, n_dof):
        raise ValueError(f'Input matrices must be of shape ({n_dof}, {n_dof}).')
    fixed_mask = np.zeros(n_dof, dtype=bool)
    if boundary_conditions is not None:
        for (node, bc) in boundary_conditions.items():
            if not isinstance(node, int) or node < 0 or node >= n_nodes:
                raise ValueError(f'Boundary condition node index {node} out of range [0, {n_nodes - 1}].')
            bc_arr = np.asarray(bc, dtype=bool).reshape(-1)
            if bc_arr.size != 6:
                raise ValueError(f'Boundary condition for node {node} must have 6 boolean entries.')
            start = node * 6
            fixed_mask[start:start + 6] = bc_arr
    free_mask = ~fixed_mask
    free_idx = np.nonzero(free_mask)[0]
    if free_idx.size == 0:
        raise ValueError('No free DOFs to solve the eigenproblem.')
    Ke_ff = K_e[np.ix_(free_idx, free_idx)]
    Kg_ff = K_g[np.ix_(free_idx, free_idx)]
    cond_tol = 1e+16
    try:
        cond_Ke = np.linalg.cond(Ke_ff)
    except Exception:
        cond_Ke = np.inf
    try:
        cond_Kg = np.linalg.cond(-Kg_ff)
    except Exception:
        cond_Kg = np.inf
    if not np.isfinite(cond_Ke) or cond_Ke > cond_tol:
        raise ValueError(f'Reduced elastic stiffness matrix is ill-conditioned/singular (cond={cond_Ke}).')
    if not np.isfinite(cond_Kg) or cond_Kg > cond_tol:
        raise ValueError(f'Reduced geometric stiffness matrix is ill-conditioned/singular (cond={cond_Kg}).')
    try:
        (eigvals, eigvecs) = scipy.linalg.eig(Ke_ff, -Kg_ff, right=True)
    except Exception as e:
        raise ValueError(f'Generalized eigenvalue solve failed: {e}')
    imag_tol = 1e-08
    real_mask = np.abs(eigvals.imag) <= imag_tol * (1.0 + np.abs(eigvals.real))
    if not np.any(real_mask):
        raise ValueError('Eigenpairs contain non-negligible complex parts; no nearly-real eigenvalues found.')
    eigvals_real = eigvals.real[real_mask]
    vecs_real = eigvecs[:, real_mask]
    pos_tol = 1e-12
    pos_mask = eigvals_real > pos_tol
    if not np.any(pos_mask):
        raise ValueError('No positive eigenvalue is found.')
    pos_vals = eigvals_real[pos_mask]
    idx_local = np.argmin(pos_vals)
    lambda_cr = float(pos_vals[idx_local])
    vec_sel = vecs_real[:, pos_mask][:, idx_local]
    vec_norm = np.linalg.norm(vec_sel)
    if vec_norm == 0 or not np.isfinite(vec_norm):
        raise ValueError('Selected eigenvector is invalid (zero or non-finite norm).')
    rel_imag = np.linalg.norm(vec_sel.imag) / vec_norm
    if rel_imag > imag_tol:
        raise ValueError('Selected eigenpair contains non-negligible complex parts.')
    phi_ff = vec_sel.real
    phi_global = np.zeros(n_dof, dtype=float)
    phi_global[free_idx] = phi_ff
    return (lambda_cr, phi_global)