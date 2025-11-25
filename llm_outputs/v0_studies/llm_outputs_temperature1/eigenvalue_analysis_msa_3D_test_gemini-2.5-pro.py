def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """

    class SimpleBC:

        def __init__(self, fixed_dofs):
            self.fixed_dofs = fixed_dofs
    n_nodes = 2
    n_dof = 6 * n_nodes
    K_e_global = np.diag(np.arange(n_dof, 0, -1, dtype=float))
    K_g_global = -np.eye(n_dof)
    expected_lambda = 1.0
    boundary_conditions = SimpleBC(fixed_dofs=[0])
    (lambda_crit, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(lambda_crit, expected_lambda)
    assert mode_shape.shape == (n_dof,)
    assert np.all(mode_shape[boundary_conditions.fixed_dofs] == 0)
    mode_idx = np.argmax(np.abs(mode_shape))
    assert mode_idx == n_dof - 1
    expected_mode_shape = np.zeros(n_dof)
    expected_mode_shape[mode_idx] = mode_shape[mode_idx]
    assert np.allclose(mode_shape, expected_mode_shape)

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned.
    """

    class SimpleBC:

        def __init__(self, fixed_dofs):
            self.fixed_dofs = fixed_dofs
    n_nodes = 1
    n_dof = 6 * n_nodes
    K_e_global = np.eye(n_dof)
    K_e_global[2, :] = K_e_global[1, :]
    K_e_global[:, 2] = K_e_global[:, 1]
    K_g_global = -np.eye(n_dof)
    boundary_conditions = SimpleBC(fixed_dofs=[0])
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """

    class SimpleBC:

        def __init__(self, fixed_dofs):
            self.fixed_dofs = fixed_dofs
    n_nodes = 1
    n_dof = 6 * n_nodes
    K_e_global = np.eye(n_dof)
    K_e_global[0:2, 0:2] = [[1, 2], [2, 1]]
    K_g_global = np.eye(n_dof)
    K_g_global[0:2, 0:2] = [[1, 0], [0, -1]]
    boundary_conditions = SimpleBC(fixed_dofs=[])
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """

    class SimpleBC:

        def __init__(self, fixed_dofs):
            self.fixed_dofs = fixed_dofs
    n_nodes = 1
    n_dof = 6 * n_nodes
    K_e_global = np.eye(n_dof)
    K_g_global = np.eye(n_dof)
    boundary_conditions = SimpleBC(fixed_dofs=[])
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """
    Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size.
    """

    class SimpleBC:

        def __init__(self, fixed_dofs):
            self.fixed_dofs = fixed_dofs
    n_nodes = 2
    n_dof = 6 * n_nodes
    K_e_global = np.diag(np.arange(n_dof, 0, -1, dtype=float))
    K_g_global_base = -np.eye(n_dof)
    boundary_conditions = SimpleBC(fixed_dofs=[0])
    (lambda_1, mode_1) = fcn(K_e_global, K_g_global_base, boundary_conditions, n_nodes)
    c = 2.5
    K_g_global_scaled = c * K_g_global_base
    (lambda_2, mode_2) = fcn(K_e_global, K_g_global_scaled, boundary_conditions, n_nodes)
    assert np.isclose(lambda_2, lambda_1 / c)
    assert mode_1.shape == (n_dof,)
    assert mode_2.shape == (n_dof,)
    norm_mode_1 = mode_1 / np.linalg.norm(mode_1)
    norm_mode_2 = mode_2 / np.linalg.norm(mode_2)
    assert np.isclose(np.abs(np.dot(norm_mode_1, norm_mode_2)), 1.0)