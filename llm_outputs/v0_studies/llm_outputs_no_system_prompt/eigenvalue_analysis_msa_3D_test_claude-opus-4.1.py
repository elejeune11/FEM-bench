def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.diag([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0])
    K_g_global = -np.eye(n_dofs)
    boundary_conditions = Mock()

    def mock_partition(bc, n):
        free_dofs = list(range(6, 12))
        return free_dofs
    import sys
    module = sys.modules[fcn.__module__]
    original_partition = getattr(module, 'partition_degrees_of_freedom', None)
    module.partition_degrees_of_freedom = mock_partition
    try:
        (lambda_cr, mode) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
        assert np.isclose(lambda_cr, 70.0, rtol=1e-10)
        assert mode.shape == (n_dofs,)
        assert np.allclose(mode[:6], 0)
        assert np.abs(mode[6]) > 0
        assert np.allclose(mode[7:], 0, atol=1e-10)
    finally:
        if original_partition is not None:
            module.partition_degrees_of_freedom = original_partition

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.zeros((n_dofs, n_dofs))
    K_e_global[0:6, 0:6] = np.eye(6)
    K_g_global = np.eye(n_dofs)
    boundary_conditions = Mock()

    def mock_partition(bc, n):
        return list(range(6, 12))
    import sys
    module = sys.modules[fcn.__module__]
    original_partition = getattr(module, 'partition_degrees_of_freedom', None)
    module.partition_degrees_of_freedom = mock_partition
    try:
        with pytest.raises(ValueError):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    finally:
        if original_partition is not None:
            module.partition_degrees_of_freedom = original_partition

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.random.randn(n_dofs, n_dofs)
    K_e_global = K_e_global + K_e_global.T + 10 * np.eye(n_dofs)
    K_g_global = np.random.randn(n_dofs, n_dofs)
    K_g_global[0, 1] = 100
    K_g_global[1, 0] = -100
    boundary_conditions = Mock()

    def mock_partition(bc, n):
        return list(range(6, 12))
    import sys
    module = sys.modules[fcn.__module__]
    original_partition = getattr(module, 'partition_degrees_of_freedom', None)
    module.partition_degrees_of_freedom = mock_partition
    try:
        with pytest.raises(ValueError):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    finally:
        if original_partition is not None:
            module.partition_degrees_of_freedom = original_partition

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.eye(n_dofs)
    K_g_global = np.eye(n_dofs)
    boundary_conditions = Mock()

    def mock_partition(bc, n):
        return list(range(6, 12))
    import sys
    module = sys.modules[fcn.__module__]
    original_partition = getattr(module, 'partition_degrees_of_freedom', None)
    module.partition_degrees_of_freedom = mock_partition
    try:
        with pytest.raises(ValueError):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    finally:
        if original_partition is not None:
            module.partition_degrees_of_freedom = original_partition

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.diag([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0])
    K_g_base = -np.eye(n_dofs)
    boundary_conditions = Mock()

    def mock_partition(bc, n):
        return list(range(6, 12))
    import sys
    module = sys.modules[fcn.__module__]
    original_partition = getattr(module, 'partition_degrees_of_freedom', None)
    module.partition_degrees_of_freedom = mock_partition
    try:
        (lambda_base, mode_base) = fcn(K_e_global, K_g_base, boundary_conditions, n_nodes)
        scale_factor = 2.0
        K_g_scaled = scale_factor * K_g_base
        (lambda_scaled, mode_scaled) = fcn(K_e_global, K_g_scaled, boundary_conditions, n_nodes)
        assert np.isclose(lambda_scaled, lambda_base / scale_factor, rtol=1e-10)
        assert mode_base.shape == (n_dofs,)
        assert mode_scaled.shape == (n_dofs,)
        assert np.allclose(mode_scaled[:6], 0)
        nonzero_base = np.abs(mode_base) > 1e-10
        nonzero_scaled = np.abs(mode_scaled) > 1e-10
        assert np.array_equal(nonzero_base, nonzero_scaled)
    finally:
        if original_partition is not None:
            module.partition_degrees_of_freedom = original_partition