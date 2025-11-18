def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""

    class MockBC:

        def __init__(self, fixed_dofs):
            self.fixed_dofs = fixed_dofs
    n_nodes = 3
    n_dofs = 6 * n_nodes
    K_e = np.diag([1.0, 3.0, 2.0, 5.0, 4.0, 1.5] + [10.0] * (n_dofs - 6))
    K_g = -np.eye(n_dofs)
    boundary_conditions = MockBC(fixed_dofs=list(range(6, n_dofs)))
    (load_factor, mode_shape) = fcn(K_e, K_g, boundary_conditions, n_nodes)
    assert np.isclose(load_factor, 1.0)
    assert mode_shape.shape == (n_dofs,)
    assert np.argmax(np.abs(mode_shape)) == 0

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""

    class MockBC:

        def __init__(self, fixed_dofs):
            self.fixed_dofs = fixed_dofs
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e = np.zeros((n_dofs, n_dofs))
    K_g = np.eye(n_dofs)
    boundary_conditions = MockBC(fixed_dofs=list(range(6, n_dofs)))
    with pytest.raises(ValueError, match='ill-conditioned|singular'):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""

    class MockBC:

        def __init__(self, fixed_dofs):
            self.fixed_dofs = fixed_dofs
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e = np.eye(n_dofs)
    K_g = np.zeros((n_dofs, n_dofs))
    K_g[0, 1] = 1.0
    K_g[1, 0] = -1.0
    boundary_conditions = MockBC(fixed_dofs=list(range(6, n_dofs)))
    with pytest.raises(ValueError, match='complex'):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""

    class MockBC:

        def __init__(self, fixed_dofs):
            self.fixed_dofs = fixed_dofs
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e = np.eye(n_dofs)
    K_g = np.eye(n_dofs)
    boundary_conditions = MockBC(fixed_dofs=list(range(6, n_dofs)))
    with pytest.raises(ValueError, match='positive'):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""

    class MockBC:

        def __init__(self, fixed_dofs):
            self.fixed_dofs = fixed_dofs
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e = np.eye(n_dofs)
    K_g_base = 0.5 * np.eye(n_dofs)
    boundary_conditions = MockBC(fixed_dofs=list(range(6, n_dofs)))
    (lambda1, mode1) = fcn(K_e, K_g_base, boundary_conditions, n_nodes)
    scaling_factor = 2.0
    (lambda2, mode2) = fcn(K_e, scaling_factor * K_g_base, boundary_conditions, n_nodes)
    assert np.isclose(lambda2, lambda1 / scaling_factor)
    assert mode1.shape == mode2.shape == (n_dofs,)
    dot_product = np.abs(np.dot(mode1 / np.linalg.norm(mode1), mode2 / np.linalg.norm(mode2)))
    assert np.isclose(dot_product, 1.0)