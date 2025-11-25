def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 2
    total_dofs = 6 * n_nodes
    constrained_dofs = list(range(6))
    bc = BoundaryConditions(constrained_dofs=constrained_dofs)
    k_diag = np.arange(1, total_dofs + 1) * 10.0
    K_e_global = np.diag(k_diag)
    K_g_global = -np.eye(total_dofs)
    expected_lambda = 70.0
    expected_mode_shape_index = 6
    (lambda_crit, mode_shape) = fcn(K_e_global, K_g_global, bc, n_nodes)
    assert np.isclose(lambda_crit, expected_lambda)
    assert mode_shape.shape == (total_dofs,)
    assert np.allclose(mode_shape[constrained_dofs], 0.0)
    assert np.argmax(np.abs(mode_shape)) == expected_mode_shape_index

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned.
    """
    n_nodes = 2
    total_dofs = 6 * n_nodes
    bc = BoundaryConditions(constrained_dofs=list(range(6)))
    K_e_global = np.eye(total_dofs)
    K_e_global[6, 6] = 0.0
    K_g_global = -np.eye(total_dofs)
    with pytest.raises(ValueError, match='(?i)ill-conditioned|singular'):
        fcn(K_e_global, K_g_global, bc, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """
    n_nodes = 2
    total_dofs = 6 * n_nodes
    bc = BoundaryConditions(constrained_dofs=list(range(6)))
    K_e_global = np.eye(total_dofs)
    K_e_global[6, 7] = 100.0
    K_e_global[7, 6] = -100.0
    K_g_global = -np.eye(total_dofs)
    with pytest.raises(ValueError, match='(?i)complex'):
        fcn(K_e_global, K_g_global, bc, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 2
    total_dofs = 6 * n_nodes
    bc = BoundaryConditions(constrained_dofs=list(range(6)))
    K_e_global = np.eye(total_dofs)
    K_g_global = np.eye(total_dofs)
    with pytest.raises(ValueError, match='(?i)No positive eigenvalue'):
        fcn(K_e_global, K_g_global, bc, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """
    Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size.
    """
    n_nodes = 2
    total_dofs = 6 * n_nodes
    bc = BoundaryConditions(constrained_dofs=list(range(6)))
    K_e_global = np.diag(np.arange(1, total_dofs + 1) * 10.0)
    K_g_global_base = -np.eye(total_dofs)
    scaling_factor = 2.5
    K_g_global_scaled = scaling_factor * K_g_global_base
    (lambda_base, mode_base) = fcn(K_e_global, K_g_global_base, bc, n_nodes)
    (lambda_scaled, mode_scaled) = fcn(K_e_global, K_g_global_scaled, bc, n_nodes)
    assert np.isclose(lambda_scaled, lambda_base / scaling_factor)
    assert mode_scaled.shape == (total_dofs,)
    norm_mode_base = mode_base / np.linalg.norm(mode_base)
    norm_mode_scaled = mode_scaled / np.linalg.norm(mode_scaled)
    assert np.allclose(np.abs(norm_mode_base), np.abs(norm_mode_scaled))