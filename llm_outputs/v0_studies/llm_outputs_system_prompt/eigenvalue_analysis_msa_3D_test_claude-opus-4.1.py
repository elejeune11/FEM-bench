def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.diag([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0])
    K_g_global = -np.eye(n_dofs)
    boundary_conditions = MagicMock()
    import eigenvalue_analysis_msa_3D
    original_partition = eigenvalue_analysis_msa_3D.partition_degrees_of_freedom
    eigenvalue_analysis_msa_3D.partition_degrees_of_freedom = lambda bc, nn: (np.array([6, 7, 8, 9, 10, 11]), np.array([0, 1, 2, 3, 4, 5]))
    try:
        (load_factor, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
        assert np.isclose(load_factor, 70.0, rtol=1e-10)
        assert np.allclose(mode_shape[:6], 0.0)
        assert mode_shape.shape == (n_dofs,)
    finally:
        eigenvalue_analysis_msa_3D.partition_degrees_of_freedom = original_partition

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.zeros((n_dofs, n_dofs))
    K_e_global[0, 0] = 1.0
    K_g_global = np.eye(n_dofs)
    boundary_conditions = MagicMock()
    import eigenvalue_analysis_msa_3D
    original_partition = eigenvalue_analysis_msa_3D.partition_degrees_of_freedom
    eigenvalue_analysis_msa_3D.partition_degrees_of_freedom = lambda bc, nn: (np.arange(n_dofs), np.array([]))
    try:
        with pytest.raises(ValueError):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    finally:
        eigenvalue_analysis_msa_3D.partition_degrees_of_freedom = original_partition

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.random.randn(n_dofs, n_dofs)
    K_e_global = K_e_global + K_e_global.T + 10 * np.eye(n_dofs)
    K_g_global = np.random.randn(n_dofs, n_dofs)
    K_g_global[0, 1] = 100.0
    K_g_global[1, 0] = -100.0
    boundary_conditions = MagicMock()
    import eigenvalue_analysis_msa_3D
    original_partition = eigenvalue_analysis_msa_3D.partition_degrees_of_freedom
    eigenvalue_analysis_msa_3D.partition_degrees_of_freedom = lambda bc, nn: (np.arange(6, n_dofs), np.arange(6))
    try:
        with pytest.raises(ValueError):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    finally:
        eigenvalue_analysis_msa_3D.partition_degrees_of_freedom = original_partition

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.eye(n_dofs)
    K_g_global = np.eye(n_dofs)
    boundary_conditions = MagicMock()
    import eigenvalue_analysis_msa_3D
    original_partition = eigenvalue_analysis_msa_3D.partition_degrees_of_freedom
    eigenvalue_analysis_msa_3D.partition_degrees_of_freedom = lambda bc, nn: (np.arange(6, n_dofs), np.arange(6))
    try:
        with pytest.raises(ValueError):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    finally:
        eigenvalue_analysis_msa_3D.partition_degrees_of_freedom = original_partition

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.diag(np.arange(1.0, n_dofs + 1.0))
    K_g_global = -np.diag(np.arange(0.1, 0.1 * (n_dofs + 1), 0.1))
    boundary_conditions = MagicMock()
    import eigenvalue_analysis_msa_3D
    original_partition = eigenvalue_analysis_msa_3D.partition_degrees_of_freedom
    eigenvalue_analysis_msa_3D.partition_degrees_of_freedom = lambda bc, nn: (np.arange(6, n_dofs), np.arange(6))
    try:
        (load_factor1, mode_shape1) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
        scaling_factor = 2.0
        K_g_scaled = scaling_factor * K_g_global
        (load_factor2, mode_shape2) = fcn(K_e_global, K_g_scaled, boundary_conditions, n_nodes)
        assert np.isclose(load_factor2, load_factor1 / scaling_factor, rtol=1e-10)
        assert mode_shape1.shape == (n_dofs,)
        assert mode_shape2.shape == (n_dofs,)
        assert np.allclose(mode_shape1[:6], 0.0)
        assert np.allclose(mode_shape2[:6], 0.0)
    finally:
        eigenvalue_analysis_msa_3D.partition_degrees_of_freedom = original_partition