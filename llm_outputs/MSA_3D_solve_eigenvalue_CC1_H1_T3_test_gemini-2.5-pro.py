def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    k_diag = np.array([100, 200, 300, 400, 500, 600, 10, 20, 30, 40, 50, 60], dtype=float)
    K_e_global = np.diag(k_diag)
    K_g_global = -np.identity(n_dofs)
    boundary_conditions = {0: [True] * 6}
    expected_lambda = 10.0
    expected_mode = np.zeros(n_dofs)
    expected_mode[6] = 1.0
    (lambda_crit, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert lambda_crit == pytest.approx(expected_lambda)
    assert mode_shape.shape == (n_dofs,)
    norm_actual = np.linalg.norm(mode_shape)
    norm_expected = np.linalg.norm(expected_mode)
    assert norm_actual > 1e-09
    dot_product = np.dot(mode_shape / norm_actual, expected_mode / norm_expected)
    assert abs(dot_product) == pytest.approx(1.0)

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 1
    n_dofs = 6 * n_nodes
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 0.0])
    K_g_global = -np.identity(n_dofs)
    boundary_conditions = {}
    with pytest.raises(ValueError, match='(?i)singular|ill-conditioned'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 1
    n_dofs = 6 * n_nodes
    K_e_global = np.diag([10.0] * n_dofs)
    K_e_global[0, 1] = 100.0
    K_e_global[1, 0] = -100.0
    K_g_global = -np.identity(n_dofs)
    boundary_conditions = {}
    with pytest.raises(ValueError, match='(?i)complex'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 1
    n_dofs = 6 * n_nodes
    K_e_global = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    K_g_global = -np.identity(n_dofs)
    boundary_conditions = {}
    with pytest.raises(ValueError, match='(?i)no positive eigenvalue'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.diag(np.arange(1, n_dofs + 1, dtype=float))
    K_g_global_base = -np.identity(n_dofs)
    boundary_conditions = {1: [True] * 6}
    (lambda_base, vec_base) = fcn(K_e_global, K_g_global_base, boundary_conditions, n_nodes)
    scaling_factor = 2.5
    K_g_global_scaled = scaling_factor * K_g_global_base
    (lambda_scaled, vec_scaled) = fcn(K_e_global, K_g_global_scaled, boundary_conditions, n_nodes)
    assert lambda_scaled == pytest.approx(lambda_base / scaling_factor)
    assert vec_scaled.shape == (n_dofs,)
    norm_base = np.linalg.norm(vec_base)
    norm_scaled = np.linalg.norm(vec_scaled)
    assert norm_base > 1e-09
    assert norm_scaled > 1e-09
    dot_product = np.dot(vec_base / norm_base, vec_scaled / norm_scaled)
    assert abs(dot_product) == pytest.approx(1.0)