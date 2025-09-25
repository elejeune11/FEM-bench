def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_global = -np.eye(dof)

    class MockBC:

        def __init__(self):
            self.constrained_dofs = list(range(6))
    boundary_conditions = MockBC()

    def mock_partition(bc, n_nodes):
        return (list(range(6, dof)), bc.constrained_dofs)
    import sys
    original_module = sys.modules[__name__]
    sys.modules[__name__].partition_degrees_of_freedom = mock_partition
    try:
        (load_factor, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
        assert abs(load_factor - 7.0) < 1e-10
        assert mode_shape.shape == (dof,)
        assert abs(mode_shape[6]) > 0.1
    finally:
        sys.modules[__name__].partition_degrees_of_freedom = original_module.partition_degrees_of_freedom

def test_eigen_singular_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.zeros((dof, dof))
    K_g_global = np.eye(dof)

    class MockBC:

        def __init__(self):
            self.constrained_dofs = []
    boundary_conditions = MockBC()

    def mock_partition(bc, n_nodes):
        return (list(range(dof)), bc.constrained_dofs)
    import sys
    original_module = sys.modules[__name__]
    sys.modules[__name__].partition_degrees_of_freedom = mock_partition
    try:
        with pytest.raises(ValueError, match='ill-conditioned/singular'):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    finally:
        sys.modules[__name__].partition_degrees_of_freedom = original_module.partition_degrees_of_freedom

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.eye(dof)
    K_g_global = np.random.rand(dof, dof)

    class MockBC:

        def __init__(self):
            self.constrained_dofs = list(range(3))
    boundary_conditions = MockBC()

    def mock_partition(bc, n_nodes):
        return (list(range(3, dof)), bc.constrained_dofs)
    import sys
    original_module = sys.modules[__name__]
    sys.modules[__name__].partition_degrees_of_freedom = mock_partition
    try:
        with pytest.raises(ValueError, match='non-negligible complex parts'):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    finally:
        sys.modules[__name__].partition_degrees_of_freedom = original_module.partition_degrees_of_freedom

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = -np.eye(dof)
    K_g_global = np.eye(dof)

    class MockBC:

        def __init__(self):
            self.constrained_dofs = list(range(3))
    boundary_conditions = MockBC()

    def mock_partition(bc, n_nodes):
        return (list(range(3, dof)), bc.constrained_dofs)
    import sys
    original_module = sys.modules[__name__]
    sys.modules[__name__].partition_degrees_of_freedom = mock_partition
    try:
        with pytest.raises(ValueError, match='no positive eigenvalue'):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    finally:
        sys.modules[__name__].partition_degrees_of_freedom = original_module.partition_degrees_of_freedom

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.diag(range(1, dof + 1))
    K_g_global_base = -np.eye(dof)

    class MockBC:

        def __init__(self):
            self.constrained_dofs = list(range(6))
    boundary_conditions = MockBC()

    def mock_partition(bc, n_nodes):
        return (list(range(6, dof)), bc.constrained_dofs)
    import sys
    original_module = sys.modules[__name__]
    sys.modules[__name__].partition_degrees_of_freedom = mock_partition
    try:
        (load_factor1, mode_shape1) = fcn(K_e_global, K_g_global_base, boundary_conditions, n_nodes)
        K_g_global_scaled = 2.0 * K_g_global_base
        (load_factor2, mode_shape2) = fcn(K_e_global, K_g_global_scaled, boundary_conditions, n_nodes)
        assert abs(load_factor2 - load_factor1 / 2.0) < 1e-10
        mode1_norm = mode_shape1 / np.linalg.norm(mode_shape1)
        mode2_norm = mode_shape2 / np.linalg.norm(mode_shape2)
        assert np.allclose(mode1_norm, mode2_norm, atol=1e-10)
        assert mode_shape1.shape == (dof,)
        assert mode_shape2.shape == (dof,)
    finally:
        sys.modules[__name__].partition_degrees_of_freedom = original_module.partition_degrees_of_freedom