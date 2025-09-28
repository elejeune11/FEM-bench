def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g = -np.eye(n_dofs)

    class MockBC:

        def __init__(self):
            self.free_dofs = list(range(n_dofs))
    boundary_conditions = MockBC()
    (load_factor, mode_shape) = fcn(K_e, K_g, boundary_conditions, n_nodes)
    expected_load_factor = 1.0
    expected_mode = np.zeros(n_dofs)
    expected_mode[0] = 1.0
    assert np.isclose(load_factor, expected_load_factor)
    assert mode_shape.shape == (n_dofs,)
    assert np.argmax(np.abs(mode_shape)) == 0

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e = np.zeros((n_dofs, n_dofs))
    K_g = -np.eye(n_dofs)

    class MockBC:

        def __init__(self):
            self.free_dofs = list(range(n_dofs))
    boundary_conditions = MockBC()
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e = np.eye(n_dofs)
    K_g = np.zeros((n_dofs, n_dofs))
    K_g[0, 1] = 1.0
    K_g[1, 0] = -1.0

    class MockBC:

        def __init__(self):
            self.free_dofs = list(range(n_dofs))
    boundary_conditions = MockBC()
    with pytest.raises(ValueError, match='complex'):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e = -np.eye(n_dofs)
    K_g = -np.eye(n_dofs)

    class MockBC:

        def __init__(self):
            self.free_dofs = list(range(n_dofs))
    boundary_conditions = MockBC()
    with pytest.raises(ValueError, match='positive'):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_base = -np.eye(n_dofs)

    class MockBC:

        def __init__(self):
            self.free_dofs = list(range(n_dofs))
    boundary_conditions = MockBC()
    (load_factor1, mode_shape1) = fcn(K_e, K_g_base, boundary_conditions, n_nodes)
    scale_factor = 2.0
    (load_factor2, mode_shape2) = fcn(K_e, scale_factor * K_g_base, boundary_conditions, n_nodes)
    assert np.isclose(load_factor2, load_factor1 / scale_factor)
    assert mode_shape1.shape == (n_dofs,)
    assert mode_shape2.shape == (n_dofs,)
    mode1_norm = mode_shape1 / np.linalg.norm(mode_shape1)
    mode2_norm = mode_shape2 / np.linalg.norm(mode_shape2)
    assert np.allclose(np.abs(mode1_norm), np.abs(mode2_norm))