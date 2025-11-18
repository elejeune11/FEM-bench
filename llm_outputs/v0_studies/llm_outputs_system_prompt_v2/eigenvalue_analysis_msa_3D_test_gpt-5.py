def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 1
    K_e_global = np.diag([2.0, 5.0, 3.0, 7.0, 11.0, 13.0])
    K_g_global = -np.eye(6)
    boundary_conditions = SimpleNamespace(constrained_dofs=np.array([], dtype=int))
    (lam, phi) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isfinite(lam)
    assert np.isclose(lam, 2.0, rtol=1e-12, atol=1e-12)
    assert isinstance(phi, np.ndarray)
    assert phi.shape == (6,)
    idx_max = int(np.argmax(np.abs(phi)))
    assert idx_max == 0
    max_abs = np.max(np.abs(phi))
    mask = np.ones(6, dtype=bool)
    mask[idx_max] = False
    assert np.all(np.abs(phi[mask]) <= 1e-09 * max_abs + 1e-12)

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned.
    """
    n_nodes = 1
    K_e_global = np.diag([2.0, 5.0, 0.0, 7.0, 11.0, 13.0])
    K_g_global = -np.eye(6)
    boundary_conditions = SimpleNamespace(constrained_dofs=np.array([], dtype=int))
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """
    n_nodes = 1
    K_e_global = np.diag([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    K_g_global = np.zeros((6, 6))
    K_g_global[0:2, 0:2] = np.array([[0.0, 1.0], [-1.0, 0.0]])
    K_g_global[2:, 2:] = -np.eye(4)
    boundary_conditions = SimpleNamespace(constrained_dofs=np.array([], dtype=int))
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 1
    K_e_global = np.diag([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    K_g_global = np.eye(6)
    boundary_conditions = SimpleNamespace(constrained_dofs=np.array([], dtype=int))
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """
    Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size.
    """
    n_nodes = 2
    ndof = 6 * n_nodes
    diag_vals = np.arange(1.0, ndof + 1.0, dtype=float)
    K_e_global = np.diag(diag_vals)
    K_g_global = -np.eye(ndof)
    constrained = np.array([0, 7], dtype=int)
    boundary_conditions = SimpleNamespace(constrained_dofs=constrained)
    (lam1, phi1) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert isinstance(phi1, np.ndarray)
    assert phi1.shape == (ndof,)
    assert np.allclose(phi1[constrained], 0.0, atol=1e-12)
    c = 4.2
    (lam2, phi2) = fcn(K_e_global, c * K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(lam2, lam1 / c, rtol=1e-10, atol=1e-12)
    assert phi2.shape == (ndof,)
    assert np.allclose(phi2[constrained], 0.0, atol=1e-12)
    free_mask = np.ones(ndof, dtype=bool)
    free_mask[constrained] = False
    phi1_free = phi1[free_mask]
    phi2_free = phi2[free_mask]
    idx_max = int(np.argmax(np.abs(phi1_free)))
    denom = phi1_free[idx_max]
    assert np.abs(denom) > 0.0
    alpha = phi2_free[idx_max] / denom
    diff_norm = np.linalg.norm(phi2_free - alpha * phi1_free)
    ref_norm = np.linalg.norm(phi2_free)
    assert diff_norm <= 1e-08 * (ref_norm + 1e-15)