def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g = -np.eye(dof)

    class MockBC:

        def __init__(self, fixed_dofs):
            self.fixed_dofs = fixed_dofs
    boundary_conditions = MockBC(fixed_dofs=list(range(1, dof)))
    (elastic_critical_load_factor, deformed_shape_vector) = fcn(K_e, K_g, boundary_conditions, n_nodes)
    assert np.isclose(elastic_critical_load_factor, 1.0)
    expected_mode = np.zeros(dof)
    expected_mode[0] = 1.0
    assert np.allclose(deformed_shape_vector, expected_mode)

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e = np.zeros((dof, dof))
    K_g = -np.eye(dof)

    class MockBC:

        def __init__(self, fixed_dofs):
            self.fixed_dofs = fixed_dofs
    boundary_conditions = MockBC(fixed_dofs=[0, 1, 2])
    with pytest.raises(ValueError, match='ill-conditioned|singular'):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e = np.eye(dof)
    K_g = np.zeros((dof, dof))
    K_g[0, 1] = 1.0
    K_g[1, 0] = -1.0

    class MockBC:

        def __init__(self, fixed_dofs):
            self.fixed_dofs = fixed_dofs
    boundary_conditions = MockBC(fixed_dofs=list(range(2, dof)))
    with pytest.raises(ValueError, match='complex'):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e = -np.eye(dof)
    K_g = -np.eye(dof)

    class MockBC:

        def __init__(self, fixed_dofs):
            self.fixed_dofs = fixed_dofs
    boundary_conditions = MockBC(fixed_dofs=[0, 1, 2])
    with pytest.raises(ValueError, match='positive'):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_base = -np.eye(dof)

    class MockBC:

        def __init__(self, fixed_dofs):
            self.fixed_dofs = fixed_dofs
    boundary_conditions = MockBC(fixed_dofs=list(range(1, dof)))
    (lambda1, mode1) = fcn(K_e, K_g_base, boundary_conditions, n_nodes)
    (lambda2, mode2) = fcn(K_e, 2.0 * K_g_base, boundary_conditions, n_nodes)
    assert np.isclose(lambda2, lambda1 / 2.0)
    assert np.allclose(np.abs(mode1), np.abs(mode2))
    assert mode1.shape == (dof,)
    assert mode2.shape == (dof,)