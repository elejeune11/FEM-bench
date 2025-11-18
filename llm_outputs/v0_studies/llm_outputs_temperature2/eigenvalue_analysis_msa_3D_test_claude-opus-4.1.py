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
        free_dofs = list(range(3, n_dofs))
        return (free_dofs, list(range(3)))
    import sys
    module = sys.modules[fcn.__module__]
    original_partition = getattr(module, 'partition_degrees_of_freedom', None)
    module.partition_degrees_of_freedom = mock_partition
    try:
        (lambda_cr, mode) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
        assert np.isclose(lambda_cr, 40.0, rtol=1e-10)
        expected_mode = np.zeros(n_dofs)
        expected_mode[3] = 1.0
        assert np.allclose(mode, expected_mode) or np.allclose(mode, -expected_mode)
    finally:
        if original_partition:
            module.partition_degrees_of_freedom = original_partition

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.ones((n_dofs, n_dofs)) * 1e-20
    K_g_global = np.eye(n_dofs)
    boundary_conditions = Mock()

    def mock_partition(bc, n):
        return (list(range(3, n_dofs)), list(range(3)))
    import sys
    module = sys.modules[fcn.__module__]
    original_partition = getattr(module, 'partition_degrees_of_freedom', None)
    module.partition_degrees_of_freedom = mock_partition
    try:
        with pytest.raises(ValueError):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    finally:
        if original_partition:
            module.partition_degrees_of_freedom = original_partition

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.array([[1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, -2, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    K_g_global = np.eye(n_dofs)
    boundary_conditions = Mock()

    def mock_partition(bc, n):
        return (list(range(3, n_dofs)), list(range(3)))
    import sys
    module = sys.modules[fcn.__module__]
    original_partition = getattr(module, 'partition_degrees_of_freedom', None)
    module.partition_degrees_of_freedom = mock_partition
    try:
        with pytest.raises(ValueError):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    finally:
        if original_partition:
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
        return (list(range(3, n_dofs)), list(range(3)))
    import sys
    module = sys.modules[fcn.__module__]
    original_partition = getattr(module, 'partition_degrees_of_freedom', None)
    module.partition_degrees_of_freedom = mock_partition
    try:
        with pytest.raises(ValueError):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    finally:
        if original_partition:
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
        return (list(range(3, n_dofs)), list(range(3)))
    import sys
    module = sys.modules[fcn.__module__]
    original_partition = getattr(module, 'partition_degrees_of_freedom', None)
    module.partition_degrees_of_freedom = mock_partition
    try:
        (lambda_base, mode_base) = fcn(K_e_global, K_g_base, boundary_conditions, n_nodes)
        scale_factor = 2.0
        K_g_scaled = K_g_base * scale_factor
        (lambda_scaled, mode_scaled) = fcn(K_e_global, K_g_scaled, boundary_conditions, n_nodes)
        assert np.isclose(lambda_scaled, lambda_base / scale_factor, rtol=1e-10)
        assert mode_scaled.shape == (n_dofs,)
        mode_base_norm = mode_base / np.linalg.norm(mode_base)
        mode_scaled_norm = mode_scaled / np.linalg.norm(mode_scaled)
        assert np.allclose(mode_base_norm, mode_scaled_norm) or np.allclose(mode_base_norm, -mode_scaled_norm)
    finally:
        if original_partition:
            module.partition_degrees_of_freedom = original_partition