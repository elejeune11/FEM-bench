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
    ix_free = np.ix_(free, free)
    K_e_ff = K_e_global[ix_free]
    K_g_ff = K_g_global[ix_free]
    if K_e_ff.shape[0] > 0 and np.linalg.cond(K_e_ff) > 1e+16:
        raise ValueError('Reduced matrices are ill-conditioned/singular beyond tolerance.')
    try:
        (eigenvalues, eigenvectors) = scipy.linalg.eig(K_e_ff, -K_g_ff)
    except (np.linalg.LinAlgError, ValueError) as e:
        raise ValueError('Eigenvalue problem failed to solve.') from e
    complex_tol = 1e-09
    if np.iscomplexobj(eigenvalues) and np.max(np.abs(eigenvalues.imag)) > complex_tol:
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    if np.iscomplexobj(eigenvectors) and np.max(np.abs(eigenvectors.imag)) > complex_tol:
        raise ValueError('Eigenpairs contain non-negligible complex parts.')
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    positive_tol = 1e-09
    positive_mask = eigenvalues > positive_tol
    if not np.any(positive_mask):
        raise ValueError('No positive eigenvalue is found.')
    eigenvalues_for_min = np.where(positive_mask, eigenvalues, np.inf)
    min_idx = np.argmin(eigenvalues_for_min)
    elastic_critical_load_factor = eigenvalues[min_idx]
    phi_free = eigenvectors[:, min_idx]
    n_dof = n_nodes * 6
    deformed_shape_vector = np.zeros(n_dof)
    if free.size > 0:
        deformed_shape_vector[free] = phi_free
    return (elastic_critical_load_factor, deformed_shape_vector)