def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple, analytically solvable case."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_global = -np.eye(dof)

    class BoundaryConditions:
        fixed_dofs = [0, 1, 2, 3, 4, 5]
    boundary_conditions = BoundaryConditions()
    (load_factor, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    expected_load_factor = 7.0
    assert np.isclose(load_factor, expected_load_factor)
    assert mode_shape.shape == (dof,)
    assert np.allclose(mode_shape[:6], 0.0)
    assert not np.allclose(mode_shape[6:], 0.0)

def test_eigen_singular_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.eye(dof)
    K_e_global[6:, 6:] = 1e-20
    K_g_global = -np.eye(dof)

    class BoundaryConditions:
        fixed_dofs = [0, 1, 2, 3, 4, 5]
    boundary_conditions = BoundaryConditions()
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.eye(dof)
    K_g_global = -np.eye(dof)
    K_e_global[6, 7] = 2.0
    K_e_global[7, 6] = -2.0

    class BoundaryConditions:
        fixed_dofs = [0, 1, 2, 3, 4, 5]
    boundary_conditions = BoundaryConditions()
    with pytest.raises(ValueError, match='complex'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive eigenvalues are present."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = -np.eye(dof)
    K_g_global = np.eye(dof)

    class BoundaryConditions:
        fixed_dofs = [0, 1, 2, 3, 4, 5]
    boundary_conditions = BoundaryConditions()
    with pytest.raises(ValueError, match='positive eigenvalue'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the reference geometric stiffness."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_global = -np.eye(dof)

    class BoundaryConditions:
        fixed_dofs = [0, 1, 2, 3, 4, 5]
    boundary_conditions = BoundaryConditions()
    (load_factor1, mode_shape1) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    scale_factor = 2.0
    K_g_scaled = scale_factor * K_g_global
    (load_factor2, mode_shape2) = fcn(K_e_global, K_g_scaled, boundary_conditions, n_nodes)
    assert np.isclose(load_factor2, load_factor1 / scale_factor)
    assert mode_shape1.shape == mode_shape2.shape == (dof,)
    assert np.allclose(mode_shape1[:6], 0.0) and np.allclose(mode_shape2[:6], 0.0)
    free_dofs = slice(6, dof)
    if not np.allclose(mode_shape1[free_dofs], 0.0):
        ratio = mode_shape2[free_dofs] / mode_shape1[free_dofs]
        non_zero = np.abs(mode_shape1[free_dofs]) > 1e-10
        if np.any(non_zero):
            scaling_factors = ratio[non_zero]
            assert np.allclose(scaling_factors, scaling_factors[0])