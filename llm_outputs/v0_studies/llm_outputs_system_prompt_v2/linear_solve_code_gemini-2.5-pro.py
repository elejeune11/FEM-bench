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
    n_dof = K_global.shape[0]
    ix_free = np.ix_(free, free)
    ix_fixed_free = np.ix_(fixed, free)
    K_ff = K_global[ix_free]
    K_sf = K_global[ix_fixed_free]
    P_f = P_global[free]
    P_s = P_global[fixed]
    if np.linalg.cond(K_ff) >= 1e+16:
        raise ValueError('The submatrix K_ff is ill-conditioned and the system cannot be reliably solved.')
    u_f = np.linalg.solve(K_ff, P_f)
    R_s = K_sf @ u_f - P_s
    u = np.zeros(n_dof)
    u[free] = u_f
    nodal_reaction_vector = np.zeros(n_dof)
    nodal_reaction_vector[fixed] = R_s
    return (u, nodal_reaction_vector)