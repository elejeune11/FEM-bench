def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
load factors reduce to the diagonal entries of K_e, so the function should
return the smallest one and a mode aligned with the corresponding DOF."""

    class MockBCs:

        def __init__(self, constrained_dofs):
            self.constrained_dofs = constrained_dofs
    n_nodes = 2
    total_dofs = 6 * n_nodes
    constrained_dofs = list(range(6))
    boundary_conditions = MockBCs(constrained_dofs=constrained_dofs)
    K_e_diag_free = [10.0, 2.0, 5.0, 8.0, 12.0, 20.0]
    K_e_global = np.diag([0.0] * 6 + K_e_diag_free)
    K_g_global = -np.eye(total_dofs)
    expected_lambda = 2.0
    expected_mode_index_in_free = 1
    (lambda_crit, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(lambda_crit, expected_lambda)
    assert mode_shape.shape == (total_dofs,)
    assert np.allclose(mode_shape[constrained_dofs], 0.0)
    free_dofs = [i for i in range(total_dofs) if i not in constrained_dofs]
    mode_free = mode_shape[free_dofs]
    assert np.argmax(np.abs(mode_free)) == expected_mode_index_in_free
    assert np.count_nonzero(np.isclose(mode_free, 0.0, atol=1e-09)) == len(mode_free) - 1

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
reduced elastic block is singular/ill-conditioned."""

    class MockBCs:

        def __init__(self, constrained_dofs):
            self.constrained_dofs = constrained_dofs
    n_nodes = 2
    total_dofs = 6 * n_nodes
    constrained_dofs = list(range(6))
    boundary_conditions = MockBCs(constrained_dofs=constrained_dofs)
    K_e_global = np.zeros((total_dofs, total_dofs))
    K_g_global = -np.eye(total_dofs)
    with pytest.raises(ValueError, match='ill-conditioned/singular'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
eigenproblem yields significantly complex pairs."""

    class MockBCs:

        def __init__(self, constrained_dofs):
            self.constrained_dofs = constrained_dofs
    n_nodes = 1
    total_dofs = 6 * n_nodes
    constrained_dofs = [2, 3, 4, 5]
    free_dofs = [0, 1]
    boundary_conditions = MockBCs(constrained_dofs=constrained_dofs)
    K_e_global = np.zeros((total_dofs, total_dofs))
    K_e_global[np.ix_(free_dofs, free_dofs)] = np.array([[0.0, 1.0], [-1.0, 0.0]])
    K_g_global = np.zeros((total_dofs, total_dofs))
    K_g_global[np.ix_(free_dofs, free_dofs)] = -np.eye(len(free_dofs))
    with pytest.raises(ValueError, match='non-negligible complex parts'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
eigenvalues are present."""

    class MockBCs:

        def __init__(self, constrained_dofs):
            self.constrained_dofs = constrained_dofs
    n_nodes = 1
    total_dofs = 6 * n_nodes
    constrained_dofs = [2, 3, 4, 5]
    free_dofs = [0, 1]
    boundary_conditions = MockBCs(constrained_dofs=constrained_dofs)
    K_e_global = np.zeros((total_dofs, total_dofs))
    K_e_global[np.ix_(free_dofs, free_dofs)] = -np.eye(len(free_dofs))
    K_g_global = np.zeros((total_dofs, total_dofs))
    K_g_global[np.ix_(free_dofs, free_dofs)] = -np.eye(len(free_dofs))
    with pytest.raises(ValueError, match='no positive eigenvalue'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
reference geometric stiffness. Scaling K_g by a constant c should scale
the reported eigenvalue by 1/c, while still returning valid global mode
vectors of the correct size."""

    class MockBCs:

        def __init__(self, constrained_dofs):
            self.constrained_dofs = constrained_dofs
    n_nodes = 2
    total_dofs = 6 * n_nodes
    constrained_dofs = list(range(6))
    boundary_conditions = MockBCs(constrained_dofs=constrained_dofs)
    K_e_global = np.diag([0.0] * 6 + [10.0, 2.0, 5.0, 8.0, 12.0, 20.0])
    K_g_global_base = -np.eye(total_dofs)
    scaling_factor = 2.5
    K_g_global_scaled = scaling_factor * K_g_global_base
    (lambda_base, mode_base) = fcn(K_e_global, K_g_global_base, boundary_conditions, n_nodes)
    (lambda_scaled, mode_scaled) = fcn(K_e_global, K_g_global_scaled, boundary_conditions, n_nodes)
    assert np.isclose(lambda_scaled, lambda_base / scaling_factor)
    assert mode_base.shape == (total_dofs,)
    assert mode_scaled.shape == (total_dofs,)
    norm_base = np.linalg.norm(mode_base)
    norm_scaled = np.linalg.norm(mode_scaled)
    assert norm_base > 1e-09
    assert norm_scaled > 1e-09
    dot_product = np.dot(mode_base, mode_scaled)
    assert np.isclose(abs(dot_product), norm_base * norm_scaled)