def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case with diagonal K_e and K_g = -I. The critical
    load factor should be the smallest diagonal entry of K_e and the mode should
    align with the corresponding DOF.
    """
    n_nodes = 1
    K_e = np.diag([5.0, 2.0, 3.0, 7.0, 1.0, 4.0])
    K_g = -np.eye(6)
    bc = {}
    lam, mode = fcn(K_e, K_g, bc, n_nodes)
    assert np.isclose(lam, 1.0, rtol=1e-10, atol=1e-12)
    assert mode.shape == (6 * n_nodes,)
    idx = np.argmax(np.abs(mode))
    assert idx == 4
    others = np.delete(np.arange(6), idx)
    maxmag = max(1.0, np.abs(mode[idx]))
    assert np.all(np.abs(mode[others]) <= 1e-10 * maxmag)

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the reduced elastic
    block is singular/ill-conditioned.
    """
    n_nodes = 1
    K_e = np.zeros((6, 6))
    K_g = -np.eye(6)
    bc = {}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, bc, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """
    n_nodes = 1
    K_e = np.zeros((6, 6))
    K_e[:2, :2] = np.array([[0.0, -2.0], [2.0, 0.0]])
    K_e[2, 2] = 3.0
    K_e[3, 3] = 4.0
    K_e[4, 4] = 5.0
    K_e[5, 5] = 6.0
    K_g = -np.eye(6)
    bc = {}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, bc, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 1
    K_e = -np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    K_g = -np.eye(6)
    bc = {}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, bc, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """
    Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size.
    """
    n_nodes = 2
    diag_node0 = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    diag_node1 = [4.0, 3.0, 2.0, 1.0, 12.0, 13.0]
    Ke_diag = np.array(diag_node0 + diag_node1, dtype=float)
    K_e = np.diag(Ke_diag)
    K_g = -np.eye(6 * n_nodes)
    bc = {0: np.array([True, True, True, True, True, True], dtype=bool)}
    lam_base, mode_base = fcn(K_e, K_g, bc, n_nodes)
    c = 4.0
    lam_scaled, mode_scaled = fcn(K_e, c * K_g, bc, n_nodes)
    assert np.isclose(lam_scaled, lam_base / c, rtol=1e-12, atol=1e-14)
    assert mode_base.shape == (6 * n_nodes,)
    assert mode_scaled.shape == (6 * n_nodes,)
    fixed = np.zeros(6 * n_nodes, dtype=bool)
    fixed[:6] = True
    assert np.allclose(mode_base[fixed], 0.0)
    assert np.allclose(mode_scaled[fixed], 0.0)
    idx_base = np.argmax(np.abs(mode_base))
    idx_scaled = np.argmax(np.abs(mode_scaled))
    assert idx_base == 9
    assert idx_scaled == 9