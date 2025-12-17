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
    fixed, free = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    n_dof_total = 6 * n_nodes
    if free.size == 0:
        raise ValueError('No free degrees of freedom to solve eigenproblem.')
    Ke = np.asarray(K_e_global)
    Kg = np.asarray(K_g_global)
    Ke_ff = Ke[np.ix_(free, free)]
    Kg_ff = Kg[np.ix_(free, free)]
    cond_tol = 1e+16
    s_Ke = scipy.linalg.svdvals(Ke_ff)
    s_Kg = scipy.linalg.svdvals(Kg_ff)
    cond_Ke = np.inf if s_Ke.size == 0 or s_Ke.min() == 0 else s_Ke.max() / s_Ke.min()
    cond_Kg = np.inf if s_Kg.size == 0 or s_Kg.min() == 0 else s_Kg.max() / s_Kg.min()
    if not np.isfinite(cond_Ke) or cond_Ke > cond_tol or (not np.isfinite(cond_Kg)) or (cond_Kg > cond_tol):
        raise ValueError('Reduced matrices are ill-conditioned or singular beyond tolerance 1e16.')
    Bneg = -Kg_ff
    try:
        eigvals, eigvecs = scipy.linalg.eig(Ke_ff, Bneg, right=True, check_finite=True)
    except Exception as e:
        raise ValueError(f'Eigenvalue computation failed: {e}')
    tol_imag = 1e-08
    finite_mask = np.isfinite(eigvals.real) & np.isfinite(eigvals.imag)
    imag_ok = np.abs(eigvals.imag) <= tol_imag * (1.0 + np.abs(eigvals.real))
    positive_real = eigvals.real > 0
    valid_mask = finite_mask & imag_ok & positive_real
    if not np.any(valid_mask):
        if np.any(finite_mask & positive_real & ~imag_ok):
            raise ValueError('Eigenpairs contain non-negligible complex parts.')
        raise ValueError('No positive eigenvalue is found.')
    valid_indices = np.where(valid_mask)[0]
    idx_min = valid_indices[np.argmin(eigvals.real[valid_mask])]
    lambda_cr = eigvals[idx_min].real
    phi_free = eigvecs[:, idx_min]
    if np.max(np.abs(phi_free.imag)) > tol_imag * (1.0 + np.max(np.abs(phi_free.real))):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    deformed_shape_vector = np.zeros(n_dof_total, dtype=float)
    deformed_shape_vector[free] = phi_free.real
    return (float(lambda_cr), deformed_shape_vector)