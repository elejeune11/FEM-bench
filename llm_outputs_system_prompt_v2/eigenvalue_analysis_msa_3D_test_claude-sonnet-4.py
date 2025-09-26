def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    total_dofs = 6 * n_nodes
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_global = -np.eye(total_dofs)
    boundary_conditions = BoundaryConditions([0, 1, 2, 3, 4, 5])
    (lambda_cr, mode) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    expected_lambda = 7.0
    assert abs(lambda_cr - expected_lambda) < 1e-12
    assert len(mode) == total_dofs
    assert np.allclose(mode[:6], 0.0)

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    total_dofs = 6 * n_nodes
    K_e_global = np.zeros((total_dofs, total_dofs))
    K_e_global[0, 0] = 1.0
    K_g_global = -np.eye(total_dofs)
    boundary_conditions = BoundaryConditions([0])
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    total_dofs = 6 * n_nodes
    K_e_global = np.eye(total_dofs)
    K_g_global = np.zeros((total_dofs, total_dofs))
    K_g_global[6, 7] = 1.0
    K_g_global[7, 6] = -1.0
    boundary_conditions = BoundaryConditions([0, 1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 2
    total_dofs = 6 * n_nodes
    K_e_global = -np.eye(total_dofs)
    K_g_global = -np.eye(total_dofs)
    boundary_conditions = BoundaryConditions([0, 1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 2
    total_dofs = 6 * n_nodes
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_global = -np.eye(total_dofs)
    boundary_conditions = BoundaryConditions([0, 1, 2, 3, 4, 5])
    (lambda_ref, mode_ref) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    c = 2.5
    K_g_scaled = c * K_g_global
    (lambda_scaled, mode_scaled) = fcn(K_e_global, K_g_scaled, boundary_conditions, n_nodes)
    assert abs(lambda_scaled - lambda_ref / c) < 1e-12
    assert len(mode_scaled) == len(mode_ref)
    assert len(mode_scaled) == total_dofs
    assert np.allclose(mode_scaled[:6], 0.0)