def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 2
    total_dofs = 6 * n_nodes
    K_e_global = np.diag([100, 200, 50, 400, 500, 600, 7, 8, 9, 10, 11, 12])
    K_g_global = -np.eye(total_dofs)
    boundary_conditions = object()
    free_dofs = list(range(6, 12))
    constrained_dofs = list(range(6))
    expected_lambda = 7.0
    expected_vector = np.zeros(total_dofs)
    expected_vector[6] = 1.0
    with patch(fcn.__module__ + '.partition_degrees_of_freedom', return_value=(free_dofs, constrained_dofs)):
        (lambda_crit, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    np.testing.assert_allclose(lambda_crit, expected_lambda)
    assert mode_shape.shape == (total_dofs,)
    dot_product = np.dot(mode_shape, expected_vector)
    norm_product = np.linalg.norm(mode_shape) * np.linalg.norm(expected_vector)
    assert norm_product > 1e-09, 'Returned mode shape is a zero vector'
    assert np.isclose(abs(dot_product / norm_product), 1.0)

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned.
    """
    n_nodes = 1
    total_dofs = 6 * n_nodes
    K_e_global = np.zeros((total_dofs, total_dofs))
    K_e_global[0:2, 0:2] = [[1, 1], [1, 1]]
    K_g_global = -np.eye(total_dofs)
    boundary_conditions = object()
    free_dofs = [0, 1]
    constrained_dofs = [2, 3, 4, 5]
    with patch(fcn.__module__ + '.partition_degrees_of_freedom', return_value=(free_dofs, constrained_dofs)):
        with pytest.raises(ValueError, match='ill-conditioned/singular'):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """
    n_nodes = 1
    total_dofs = 6 * n_nodes
    K_e_global = np.zeros((total_dofs, total_dofs))
    K_e_global[0:2, 0:2] = [[0, -1], [1, 0]]
    K_g_global = -np.eye(total_dofs)
    boundary_conditions = object()
    free_dofs = [0, 1]
    constrained_dofs = [2, 3, 4, 5]
    with patch(fcn.__module__ + '.partition_degrees_of_freedom', return_value=(free_dofs, constrained_dofs)):
        with pytest.raises(ValueError, match='non-negligible complex parts'):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 1
    total_dofs = 6 * n_nodes
    K_e_global = -np.eye(total_dofs)
    K_g_global = -np.eye(total_dofs)
    boundary_conditions = object()
    free_dofs = list(range(total_dofs))
    constrained_dofs = []
    with patch(fcn.__module__ + '.partition_degrees_of_freedom', return_value=(free_dofs, constrained_dofs)):
        with pytest.raises(ValueError, match='no positive eigenvalue'):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """
    Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size.
    """
    n_nodes = 2
    total_dofs = 6 * n_nodes
    K_e_global = np.diag(np.arange(1, total_dofs + 1))
    K_g_global_base = -np.eye(total_dofs)
    boundary_conditions = object()
    free_dofs = list(range(6, 12))
    constrained_dofs = list(range(6))
    scaling_factor = 2.5
    K_g_global_scaled = scaling_factor * K_g_global_base
    with patch(fcn.__module__ + '.partition_degrees_of_freedom', return_value=(free_dofs, constrained_dofs)):
        (lambda_base, vec_base) = fcn(K_e_global, K_g_global_base, boundary_conditions, n_nodes)
        (lambda_scaled, vec_scaled) = fcn(K_e_global, K_g_global_scaled, boundary_conditions, n_nodes)
    np.testing.assert_allclose(lambda_scaled, lambda_base / scaling_factor)
    assert vec_scaled.shape == (total_dofs,)
    dot_product = np.dot(vec_base, vec_scaled)
    norm_product = np.linalg.norm(vec_base) * np.linalg.norm(vec_scaled)
    assert norm_product > 1e-09, 'Returned mode shape is a zero vector'
    assert np.isclose(abs(dot_product / norm_product), 1.0)