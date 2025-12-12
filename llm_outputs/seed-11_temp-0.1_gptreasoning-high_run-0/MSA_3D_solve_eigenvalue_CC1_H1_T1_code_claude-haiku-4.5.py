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
    cond_number = np.linalg.cond(K_e_ff)
    if cond_number > 1e+16:
        raise ValueError(f'Reduced elastic stiffness matrix is ill-conditioned (condition number: {cond_number})')
    cond_number_g = np.linalg.cond(K_g_ff)
    if cond_number_g > 1e+16:
        raise ValueError(f'Reduced geometric stiffness matrix is ill-conditioned (condition number: {cond_number_g})')
    (eigenvalues, eigenvectors) = scipy.linalg.eigh(K_e_ff, -K_g_ff)
    if np.any(np.abs(eigenvalues.imag) > 1e-10):
        raise ValueError('Eigenpairs contain non-negligible complex parts')
    eigenvalues = eigenvalues.real
    positive_mask = eigenvalues > 1e-10
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalue found')
    positive_eigenvalues = eigenvalues[positive_mask]
    positive_indices = np.where(positive_mask)[0]
    min_idx = positive_indices[np.argmin(positive_eigenvalues)]
    elastic_critical_load_factor = eigenvalues[min_idx]
    mode_vector_free = eigenvectors[:, min_idx].real
    deformed_shape_vector = np.zeros(6 * n_nodes)
    deformed_shape_vector[free] = mode_vector_free
    return (elastic_critical_load_factor, deformed_shape_vector)