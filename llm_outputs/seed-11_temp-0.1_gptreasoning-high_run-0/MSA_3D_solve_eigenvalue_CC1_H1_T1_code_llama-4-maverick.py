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
    if np.linalg.cond(K_e_ff) > 1e+16 or np.linalg.cond(K_g_ff) > 1e+16:
        raise ValueError('Reduced matrices are ill-conditioned/singular beyond tolerance.')
    (eigenvalues, eigenvectors) = scipy.linalg.eig(K_e_ff, -K_g_ff)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    if np.any(np.abs(np.imag(eigenvalues)) > 1e-12) or np.any(np.abs(np.imag(eigenvectors)) > 1e-12):
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    if len(positive_eigenvalues) == 0:
        raise ValueError('No positive eigenvalue found.')
    idx_min_positive_eigenvalue = np.argmin(positive_eigenvalues)
    elastic_critical_load_factor = positive_eigenvalues[idx_min_positive_eigenvalue]
    deformed_shape_vector_free = eigenvectors[:, idx_min_positive_eigenvalue]
    deformed_shape_vector = np.zeros(6 * n_nodes)
    deformed_shape_vector[free] = deformed_shape_vector_free
    return (elastic_critical_load_factor, deformed_shape_vector)