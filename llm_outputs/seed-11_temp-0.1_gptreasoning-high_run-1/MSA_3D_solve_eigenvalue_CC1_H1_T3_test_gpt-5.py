def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result for a simple diagonal case.
    With K_e diagonal and K_g = -I, the eigenvalues are the diagonal entries of K_e.
    The function should return the smallest positive eigenvalue and a mode aligned with that DOF.
    """
    n_nodes = 1
    K_e_diag = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    K_e_global = np.diag(K_e_diag)
    K_g_global = -np.eye(6)
    boundary_conditions = {}
    lam, mode = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isfinite(lam)
    assert abs(lam - 2.0) < 1e-10
    assert mode.shape == (6 * n_nodes,)
    max_idx = int(np.argmax(np.abs(mode)))
    assert max_idx == 0
    assert abs(mode[0]) > 0
    if np.linalg.norm(mode) > 0:
        other = np.delete(mode, 0)
        assert np.linalg.norm(other) <= 1e-08 * abs(mode[0])

def test_eigen_singluar_detected(fcn):
    """
    Verify that a ValueError is raised when the reduced elastic stiffness block is singular or ill-conditioned.
    """
    n_nodes = 1
    K_e_global = np.zeros((6, 6))
    K_g_global = -np.eye(6)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that a ValueError is raised when the generalized eigenproblem yields significantly complex eigenpairs.
    Construct a 2x2 skew-symmetric K_e block (with K_g = -I), which has purely imaginary eigenvalues.
    """
    n_nodes = 1
    K_e_global = np.zeros((6, 6))
    K_e_global[0:2, 0:2] = np.array([[0.0, -1.0], [1.0, 0.0]])
    K_e_global[2, 2] = 10.0
    K_e_global[3, 3] = 20.0
    K_e_global[4, 4] = 30.0
    K_e_global[5, 5] = 40.0
    K_g_global = -np.eye(6)
    boundary_conditions = {0: np.array([False, False, True, True, True, True], dtype=bool)}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that a ValueError is raised when there are no positive eigenvalues.
    Use K_g = +I so that eigenvalues are negative of those from K_e, making them all non-positive.
    """
    n_nodes = 1
    K_e_global = np.zeros((6, 6))
    K_e_global[0, 0] = 1.0
    K_e_global[1, 1] = 2.0
    K_e_global[2, 2] = 10.0
    K_e_global[3, 3] = 20.0
    K_e_global[4, 4] = 30.0
    K_e_global[5, 5] = 40.0
    K_g_global = np.eye(6)
    boundary_conditions = {0: np.array([False, False, True, True, True, True], dtype=bool)}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """
    Check scaling invariance: scaling K_g by c should scale the reported eigenvalue by 1/c.
    Also check that the returned mode vector has correct global size.
    """
    n_nodes = 1
    K_e_diag = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    K_e_global = np.diag(K_e_diag)
    K_g_base = -np.eye(6)
    boundary_conditions = {}
    lam_base, mode_base = fcn(K_e_global, K_g_base, boundary_conditions, n_nodes)
    assert mode_base.shape == (6 * n_nodes,)
    assert np.isfinite(lam_base)
    c = 4.2
    K_g_scaled = c * K_g_base
    lam_scaled, mode_scaled = fcn(K_e_global, K_g_scaled, boundary_conditions, n_nodes)
    assert mode_scaled.shape == (6 * n_nodes,)
    assert np.isfinite(lam_scaled)
    assert abs(lam_scaled - lam_base / c) < 1e-10