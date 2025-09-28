def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""

    class MockBC:

        def __init__(self):
            self.free_dofs = [0, 1, 2]
            self.fixed_dofs = []
    n_nodes = 1
    K_e_global = np.diag([3.0, 2.0, 1.0])
    K_g_global = -np.eye(3)
    boundary_conditions = MockBC()
    (load_factor, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    expected_load_factor = 1.0
    expected_mode_shape = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])[:6 * n_nodes]
    assert np.isclose(load_factor, expected_load_factor)
    assert np.allclose(mode_shape, expected_mode_shape)

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""

    class MockBC:

        def __init__(self):
            self.free_dofs = [0, 1]
            self.fixed_dofs = [2, 3, 4, 5]
    n_nodes = 1
    K_e_global = np.zeros((6, 6))
    K_g_global = np.eye(6)
    boundary_conditions = MockBC()
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""

    class MockBC:

        def __init__(self):
            self.free_dofs = [0, 1]
            self.fixed_dofs = [2, 3, 4, 5]
    n_nodes = 1
    K_e_global = np.eye(6)
    K_g_global = np.array([[1, -2], [2, 1]])
    K_g_global = np.block([[K_g_global, np.zeros((2, 4))], [np.zeros((4, 2)), np.eye(4)]])
    boundary_conditions = MockBC()
    with pytest.raises(ValueError, match='complex'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""

    class MockBC:

        def __init__(self):
            self.free_dofs = [0, 1]
            self.fixed_dofs = [2, 3, 4, 5]
    n_nodes = 1
    K_e_global = -np.eye(6)
    K_g_global = np.eye(6)
    boundary_conditions = MockBC()
    with pytest.raises(ValueError, match='positive'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""

    class MockBC:

        def __init__(self):
            self.free_dofs = [0, 1, 2]
            self.fixed_dofs = [3, 4, 5]
    n_nodes = 1
    K_e_global = np.diag([3.0, 2.0, 1.0, 4.0, 5.0, 6.0])
    K_g_global = -np.eye(6)
    boundary_conditions = MockBC()
    scaling_factor = 2.0
    K_g_scaled = scaling_factor * K_g_global
    (load_factor1, mode_shape1) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    (load_factor2, mode_shape2) = fcn(K_e_global, K_g_scaled, boundary_conditions, n_nodes)
    assert np.isclose(load_factor2, load_factor1 / scaling_factor)
    assert mode_shape1.shape == mode_shape2.shape == (6 * n_nodes,)
    assert np.allclose(np.abs(mode_shape1), np.abs(mode_shape2))