def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_global = -np.eye(dof)

    class MockBC:

        def __getattr__(self, name):
            return None
    boundary_conditions = MockBC()
    original_partition = None
    try:
        from module_name import partition_degrees_of_freedom
        original_partition = partition_degrees_of_freedom
    except ImportError:
        pass

    def mock_partition(bc, n_nodes):
        fixed = list(range(1, 6 * n_nodes))
        free = [0]
        return (free, fixed)
    import sys
    current_module = sys.modules[__name__]
    setattr(current_module, 'partition_degrees_of_freedom', mock_partition)
    try:
        (load_factor, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
        assert load_factor == pytest.approx(1.0)
        expected_mode = np.zeros(dof)
        expected_mode[0] = 1.0
        np.testing.assert_array_equal(mode_shape, expected_mode)
    finally:
        if original_partition is not None:
            setattr(current_module, 'partition_degrees_of_freedom', original_partition)

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.zeros((dof, dof))
    K_g_global = -np.eye(dof)

    class MockBC:

        def __getattr__(self, name):
            return None
    boundary_conditions = MockBC()
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.eye(dof)
    K_g_global = np.random.rand(dof, dof)

    class MockBC:

        def __getattr__(self, name):
            return None
    boundary_conditions = MockBC()
    original_eigh = np.linalg.eigh

    def mock_eigh(A, B):
        eigvals = np.array([1 + 1j, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        eigvecs = np.eye(len(eigvals))
        return (eigvals, eigvecs)
    np.linalg.eigh = mock_eigh
    try:
        with pytest.raises(ValueError, match='complex'):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    finally:
        np.linalg.eigh = original_eigh

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.eye(dof)
    K_g_global = np.eye(dof)

    class MockBC:

        def __getattr__(self, name):
            return None
    boundary_conditions = MockBC()
    original_eigh = np.linalg.eigh

    def mock_eigh(A, B):
        eigvals = np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12])
        eigvecs = np.eye(len(eigvals))
        return (eigvals, eigvecs)
    np.linalg.eigh = mock_eigh
    try:
        with pytest.raises(ValueError, match='positive'):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    finally:
        np.linalg.eigh = original_eigh

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_global_base = -np.eye(dof)

    class MockBC:

        def __getattr__(self, name):
            return None
    boundary_conditions = MockBC()
    original_partition = None
    try:
        from module_name import partition_degrees_of_freedom
        original_partition = partition_degrees_of_freedom
    except ImportError:
        pass

    def mock_partition(bc, n_nodes):
        fixed = list(range(1, 6 * n_nodes))
        free = [0]
        return (free, fixed)
    import sys
    current_module = sys.modules[__name__]
    setattr(current_module, 'partition_degrees_of_freedom', mock_partition)
    try:
        (load_factor1, mode_shape1) = fcn(K_e_global, K_g_global_base, boundary_conditions, n_nodes)
        scaling_factor = 2.0
        K_g_global_scaled = scaling_factor * K_g_global_base
        (load_factor2, mode_shape2) = fcn(K_e_global, K_g_global_scaled, boundary_conditions, n_nodes)
        assert load_factor2 == pytest.approx(load_factor1 / scaling_factor)
        np.testing.assert_array_equal(mode_shape1, mode_shape2)
        assert mode_shape1.shape == (dof,)
        assert mode_shape2.shape == (dof,)
    finally:
        if original_partition is not None:
            setattr(current_module, 'partition_degrees_of_freedom', original_partition)