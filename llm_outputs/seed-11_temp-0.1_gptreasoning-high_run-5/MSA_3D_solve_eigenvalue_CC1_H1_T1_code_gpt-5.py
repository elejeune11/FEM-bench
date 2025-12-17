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
    n_dof = int(6 * n_nodes)
    K_e = np.asarray(K_e_global, dtype=float)
    K_g = np.asarray(K_g_global, dtype=float)
    if K_e.shape != (n_dof, n_dof) or K_g.shape != (n_dof, n_dof):
        raise ValueError('Input matrices must have shape (6*n_nodes, 6*n_nodes).')
    fixed, free = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    if free.size == 0:
        raise ValueError('No free DOFs available for buckling analysis.')
    K_e_ff = K_e[np.ix_(free, free)]
    K_g_ff = K_g[np.ix_(free, free)]
    cond_tol = 1e+16

    def _cond_ok(M):
        try:
            c = np.linalg.cond(M)
        except Exception:
            return False
        if not np.isfinite(c):
            return False
        return c <= cond_tol
    if not _cond_ok(K_e_ff) or not _cond_ok(K_g_ff):
        raise ValueError('Reduced matrices are ill-conditioned/singular beyond tolerance.')
    try:
        evals, evecs = scipy.linalg.eig(K_e_ff, -K_g_ff)
    except Exception as exc:
        raise ValueError(f'Eigenvalue solver failed: {exc}')
    imag_tol = 1e-08
    pos_tol = 1e-12
    if evals.size == 0:
        raise ValueError('No eigenvalues returned by solver.')
    finite_mask = np.isfinite(evals)
    evals_f = evals[finite_mask]
    evecs_f = evecs[:, finite_mask] if evecs is not None else None
    if evals_f.size == 0:
        raise ValueError('No finite eigenvalues found.')
    real_mask = np.abs(evals_f.imag) <= imag_tol * (1.0 + np.abs(evals_f.real))
    evals_real = evals_f.real[real_mask]
    evecs_real = evecs_f[:, real_mask] if evecs_f is not None else None
    pos_mask = evals_real > pos_tol
    if not np.any(pos_mask):
        raise ValueError('No positive eigenvalue is found.')
    evals_pos = evals_real[pos_mask]
    evecs_pos = evecs_real[:, pos_mask]
    idx_min = np.argmin(evals_pos)
    lambda_cr = float(evals_pos[idx_min])
    phi_ff = evecs_pos[:, idx_min]
    if np.abs(phi_ff.imag).max(initial=0.0) > imag_tol * (1.0 + np.abs(phi_ff.real).max(initial=0.0)):
        raise ValueError('Selected eigenvector contains non-negligible complex parts.')
    if np.abs(lambda_cr) == 0.0 or not np.isfinite(lambda_cr):
        raise ValueError('Selected eigenvalue is invalid.')
    phi_global = np.zeros(n_dof, dtype=float)
    phi_global[free] = phi_ff.real
    return (lambda_cr, phi_global)