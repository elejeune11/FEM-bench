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
    fixed = np.asarray(fixed)
    free = np.asarray(free)
    K_ff = K_global[np.ix_(free, free)]
    P_f = P_global[free]
    cond = np.linalg.cond(K_ff)
    if cond >= 1e+16:
        raise ValueError('Ill-conditioned matrix: cond(K_ff) = {:.2e}'.format(cond))
    u_f = np.linalg.solve(K_ff, P_f)
    u = np.zeros_like(P_global)
    u[free] = u_f
    nodal_reaction_vector = np.zeros_like(P_global)
    nodal_reaction_vector[fixed] = K_global[np.ix_(fixed, free)] @ u_f - P_global[fixed]
    return (u, nodal_reaction_vector)