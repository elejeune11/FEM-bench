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
    P_global = np.asarray(P_global).reshape(-1)
    K_global = np.asarray(K_global)
    fixed = np.asarray(fixed, dtype=int)
    free = np.asarray(free, dtype=int)
    n_dof = P_global.size
    u = np.zeros(n_dof, dtype=float)
    nodal_reaction_vector = np.zeros(n_dof, dtype=float)
    if free.size > 0:
        K_ff = K_global[np.ix_(free, free)]
        P_f = P_global[free]
        cond_val = np.linalg.cond(K_ff)
        if not np.isfinite(cond_val) or cond_val >= 1e+16:
            raise ValueError('Submatrix K_ff is ill-conditioned; cannot reliably solve the system.')
        u_f = np.linalg.solve(K_ff, P_f)
        u[free] = u_f
    else:
        u_f = np.array([], dtype=float)
    if fixed.size > 0:
        K_cf = K_global[np.ix_(fixed, free)] if free.size > 0 else np.zeros((fixed.size, 0), dtype=K_global.dtype)
        P_c = P_global[fixed]
        R_c = K_cf.dot(u_f) - P_c
        nodal_reaction_vector[fixed] = R_c
    return (u, nodal_reaction_vector)