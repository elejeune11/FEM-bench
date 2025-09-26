def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.diag([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0])
    K_g_global = -np.eye(n_dofs)

    class BoundaryConditions:

        def __init__(self):
            self.constrained_dofs = list(range(6))
    boundary_conditions = BoundaryConditions()
    (lambda_cr, mode) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(lambda_cr, 70.0, rtol=1e-10)
    assert mode.shape == (n_dofs,)
    assert np.all(np.abs(mode[:6]) < 1e-10)
    free_mode = mode[6:12]
    assert np.abs(free_mode[0]) > 1e-06
    for i in range(1, 6):
        assert np.abs(free_mode[i] / free_mode[0]) < 1e-06

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.zeros((n_dofs, n_dofs))
    K_e_global[:6, :6] = np.eye(6)
    K_g_global = -np.eye(n_dofs)

    class BoundaryConditions:

        def __init__(self):
            self.constrained_dofs = list(range(6))
    boundary_conditions = BoundaryConditions()
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.eye(n_dofs)
    K_e_global[6:12, 6:12] = np.array([[1, 10, 0, 0, 0, 0], [-10, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
    K_g_global = -np.eye(n_dofs)

    class BoundaryConditions:

        def __init__(self):
            self.constrained_dofs = list(range(6))
    boundary_conditions = BoundaryConditions()
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.eye(n_dofs)
    K_g_global = np.eye(n_dofs)

    class BoundaryConditions:

        def __init__(self):
            self.constrained_dofs = list(range(6))
    boundary_conditions = BoundaryConditions()
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 3
    n_dofs = 6 * n_nodes
    np.random.seed(42)
    A = np.random.randn(n_dofs, n_dofs)
    K_e_global = A.T @ A + np.eye(n_dofs)
    B = np.random.randn(n_dofs, n_dofs)
    K_g_base = -(B.T @ B + 0.1 * np.eye(n_dofs))

    class BoundaryConditions:

        def __init__(self):
            self.constrained_dofs = list(range(6))
    boundary_conditions = BoundaryConditions()
    (lambda_1, mode_1) = fcn(K_e_global, K_g_base, boundary_conditions, n_nodes)
    scale_factor = 2.0
    K_g_scaled = scale_factor * K_g_base
    (lambda_2, mode_2) = fcn(K_e_global, K_g_scaled, boundary_conditions, n_nodes)
    assert np.isclose(lambda_2, lambda_1 / scale_factor, rtol=1e-10)
    assert mode_1.shape == (n_dofs,)
    assert mode_2.shape == (n_dofs,)
    if np.linalg.norm(mode_1) > 1e-10 and np.linalg.norm(mode_2) > 1e-10:
        mode_1_norm = mode_1 / np.linalg.norm(mode_1)
        mode_2_norm = mode_2 / np.linalg.norm(mode_2)
        dot_product = np.abs(np.dot(mode_1_norm, mode_2_norm))
        assert np.isclose(dot_product, 1.0, rtol=1e-06)