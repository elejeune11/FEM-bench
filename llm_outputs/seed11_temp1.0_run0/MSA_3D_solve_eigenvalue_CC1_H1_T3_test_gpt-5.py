def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 1
    K_e = np.diag([3.0, 5.0, 2.0, 7.0, 11.0, 13.0])
    K_g = -np.eye(6)
    bc = {}
    (lam, vec) = fcn(K_e, K_g, bc, n_nodes)
    assert abs(lam - 2.0) < 1e-10
    assert vec.shape == (6 * n_nodes,)
    idx = int(np.argmax(np.abs(vec)))
    assert idx == 2
    amp = np.abs(vec[idx])
    assert amp > 0
    mask = np.ones(6, dtype=bool)
    mask[idx] = False
    assert np.linalg.norm(vec[mask]) <= 1e-10 * max(1.0, amp)

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned.
    """
    n_nodes = 1
    K_e = np.diag([3.0, 5.0, 2.0, 7.0, 11.0, 13.0])
    K_g = np.zeros((6, 6))
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
    K_e[0, 1] = -1.0
    K_e[1, 0] = 1.0
    K_g = -np.eye(6)
    bc = {0: np.array([False, False, True, True, True, True], dtype=bool)}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, bc, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 1
    K_e = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
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
    n_nodes = 1
    diag_vals = np.array([4.0, 9.0, 2.0, 5.0, 8.0, 1.0])
    K_e = np.diag(diag_vals)
    K_g0 = -np.eye(6)
    bc = {}
    (lam1, vec1) = fcn(K_e, K_g0, bc, n_nodes)
    assert vec1.shape == (6 * n_nodes,)
    idx1 = int(np.argmax(np.abs(vec1)))
    assert idx1 == int(np.argmin(diag_vals))
    c = 3.7
    K_g_scaled = c * K_g0
    (lam2, vec2) = fcn(K_e, K_g_scaled, bc, n_nodes)
    assert abs(lam2 - lam1 / c) < 1e-10
    assert vec2.shape == (6 * n_nodes,)
    idx2 = int(np.argmax(np.abs(vec2)))
    assert idx2 == idx1