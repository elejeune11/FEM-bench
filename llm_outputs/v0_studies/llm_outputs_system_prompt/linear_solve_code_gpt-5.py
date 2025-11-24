def linear_solve(P_global, K_global, fixed, free):
    """
    Solves the linear system for displacements and internal nodal forces in a 3D linear elastic structure,
    using a partitioned approach based on fixed and free degrees of freedom (DOFs).
    The function solves for displacements at the free DOFs by inverting the corresponding submatrix
    of the global stiffness matrix (`K_ff`). A condition number check (`cond(K_ff) < 1e16`) is used
    to ensure numerical stability. If the matrix is well-conditioned, the system is solved and a nodal
    reaction vector is computed at the fixed DOFs.
    Parameters
    ----------
    P_global : ndarray of shape (n_dof,)
        The global load vector.
    K_global : ndarray of shape (n_dof, n_dof)
        The global stiffness matrix.
    fixed : array-like of int
        Indices of fixed degrees of freedom.
    free : array-like of int
        Indices of free degrees of freedom.
    Returns
    -------
    u : ndarray of shape (n_dof,)
        Displacement vector. Displacements are computed only for free DOFs; fixed DOFs are set to zero.
    nodal_reaction_vector : ndarray of shape (n_dof,)
        Nodal reaction vector. Reactions are computed only for fixed DOFs.
    Raises
    ------
    ValueError
        If the submatrix `K_ff` is ill-conditioned and the system cannot be reliably solved.
    """
    import numpy as np
    K = np.asarray(K_global, dtype=float)
    P = np.asarray(P_global, dtype=float).reshape(-1)
    n = K.shape[0]
    u = np.zeros(n, dtype=float)
    nodal_reaction_vector = np.zeros(n, dtype=float)
    fixed_idx = np.asarray(fixed, dtype=int).ravel()
    free_idx = np.asarray(free, dtype=int).ravel()
    if free_idx.size > 0:
        K_ff = K[np.ix_(free_idx, free_idx)]
        cond_val = np.linalg.cond(K_ff)
        if not np.isfinite(cond_val) or cond_val >= 1e+16:
            raise ValueError('K_ff is ill-conditioned; the system cannot be reliably solved.')
        P_f = P[free_idx]
        u_f = np.linalg.solve(K_ff, P_f)
        u[free_idx] = u_f
    if fixed_idx.size > 0:
        K_cf = K[np.ix_(fixed_idx, free_idx)] if free_idx.size > 0 else np.zeros((fixed_idx.size, 0), dtype=float)
        P_c = P[fixed_idx]
        u_f = u[free_idx] if free_idx.size > 0 else np.zeros(0, dtype=float)
        reactions_fixed = K_cf.dot(u_f) - P_c
        nodal_reaction_vector[fixed_idx] = reactions_fixed
    return (u, nodal_reaction_vector)