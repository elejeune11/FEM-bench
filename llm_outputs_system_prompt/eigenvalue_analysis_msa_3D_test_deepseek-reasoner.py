def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case."""
    n_nodes = 2
    dofs = 6 * n_nodes
    K_e_global = np.diag([10.0, 5.0, 8.0, 3.0, 12.0, 6.0, 9.0, 4.0, 7.0, 2.0, 11.0, 1.0])
    K_g_global = -np.eye(dofs)

    class SimpleBC:

        def __init__(self):
            self.fixed_dofs = list(range(1, dofs))
    boundary_conditions = SimpleBC()
    (critical_load, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert critical_load == pytest.approx(1.0, rel=1e-10)
    expected_mode = np.zeros(dofs)
    expected_mode[11] = 1.0
    np.testing.assert_allclose(mode_shape, expected_mode, atol=1e-10)

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    dofs = 6 * n_nodes
    K_e_global = np.zeros((dofs, dofs))
    K_g_global = np.eye(dofs)

    class SimpleBC:

        def __init__(self):
            self.fixed_dofs = []
    boundary_conditions = SimpleBC()
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    dofs = 6 * n_nodes
    K_e_global = np.random.rand(dofs, dofs)
    K_g_global = np.random.rand(dofs, dofs)

    class SimpleBC:

        def __init__(self):
            self.fixed_dofs = [0, 1, 2, 3, 4, 5]
    boundary_conditions = SimpleBC()
    with pytest.raises(ValueError, match='complex'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 2
    dofs = 6 * n_nodes
    K_e_global = -np.diag(np.arange(1, dofs + 1))
    K_g_global = np.eye(dofs)

    class SimpleBC:

        def __init__(self):
            self.fixed_dofs = [0, 1, 2, 3, 4, 5]
    boundary_conditions = SimpleBC()
    with pytest.raises(ValueError, match='positive'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness."""
    n_nodes = 2
    dofs = 6 * n_nodes
    K_e_global = np.diag(np.arange(1, dofs + 1))
    K_g_global = -np.eye(dofs)

    class SimpleBC:

        def __init__(self):
            self.fixed_dofs = [0, 1, 2, 3, 4, 5]
    boundary_conditions = SimpleBC()
    (critical_load_1, mode_shape_1) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    scaling_factor = 2.0
    (critical_load_2, mode_shape_2) = fcn(K_e_global, scaling_factor * K_g_global, boundary_conditions, n_nodes)
    assert critical_load_2 == pytest.approx(critical_load_1 / scaling_factor, rel=1e-10)
    dot_product = np.abs(np.dot(mode_shape_1, mode_shape_2) / (np.linalg.norm(mode_shape_1) * np.linalg.norm(mode_shape_2)))
    assert dot_product == pytest.approx(1.0, rel=1e-10)