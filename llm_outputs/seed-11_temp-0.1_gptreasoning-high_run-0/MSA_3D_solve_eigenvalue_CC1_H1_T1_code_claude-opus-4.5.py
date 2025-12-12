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
    K_e_ff = K_e_global[np.ix_(free, free)]
    K_g_ff = K_g_global[np.ix_(free, free)]
    cond_K_e = np.linalg.cond(K_e_ff)
    if cond_K_e > 1e+16:
        raise ValueError('Reduced elastic stiffness matrix is ill-conditioned/singular beyond tolerance.')
    cond_K_g = np.linalg.cond(K_g_ff)
    if cond_K_g > 1e+16:
        raise ValueError('Reduced geometric stiffness matrix is ill-conditioned/singular beyond tolerance.')
    B_ff = -K_g_ff
    (eigenvalues, eigenvectors) = scipy.linalg.eig(K_e_ff, B_ff)
    if np.any(np.abs(eigenvalues.imag) > 1e-10 * np.abs(eigenvalues.real + 1e-30)):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    eigenvalues_real = eigenvalues.real
    positive_mask = eigenvalues_real > 0
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalue is found.')
    positive_eigenvalues = eigenvalues_real[positive_mask]
    positive_eigenvectors = eigenvectors[:, positive_mask]
    min_idx = np.argmin(positive_eigenvalues)
    elastic_critical_load_factor = positive_eigenvalues[min_idx]
    phi_free = positive_eigenvectors[:, min_idx]
    if np.any(np.abs(phi_free.imag) > 1e-10 * np.abs(phi_free.real + 1e-30)):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    phi_free = phi_free.real
    n_dof = 6 * n_nodes
    deformed_shape_vector = np.zeros(n_dof)
    deformed_shape_vector[free] = phi_free
    return (float(elastic_critical_load_factor), deformed_shape_vector)