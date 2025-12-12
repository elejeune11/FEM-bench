def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e; the function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 1
    size = 6 * n_nodes
    diag_vals = np.array([3.0, 5.0, 2.0, 7.0, 11.0, 13.0])
    K_e = np.diag(diag_vals)
    K_g = -np.eye(size)
    boundary_conditions = {}
    (lam, mode) = fcn(K_e, K_g, boundary_conditions, n_nodes)
    assert np.isclose(lam, 2.0, rtol=1e-08, atol=1e-12)
    assert mode.shape == (size,)
    idx_max = int(np.argmax(np.abs(mode)))
    assert idx_max == 2
    others = np.delete(mode, idx_max)
    assert np.linalg.norm(others, ord=2) <= 1e-08 * max(1.0, abs(mode[idx_max]))

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the reduced elastic
    block is singular/ill-conditioned.
    """
    n_nodes = 1
    size = 6 * n_nodes
    K_e = np.zeros((size, size))
    K_g = -np.eye(size)
    boundary_conditions = {0: np.array([False, False, True, True, True, True], dtype=bool)}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex eigenpairs.
    """
    n_nodes = 1
    size = 6 * n_nodes
    K_e = np.zeros((size, size))
    K_e[:2, :2] = np.array([[0.0, -1.0], [1.0, 0.0]])
    K_g = np.eye(size)
    boundary_conditions = {0: np.array([False, False, True, True, True, True], dtype=bool)}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 1
    size = 6 * n_nodes
    K_e = np.zeros((size, size))
    K_e[0, 0] = -3.0
    K_e[1, 1] = -2.0
    K_g = -np.eye(size)
    boundary_conditions = {0: np.array([False, False, True, True, True, True], dtype=bool)}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """
    Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size.
    """
    n_nodes = 1
    size = 6 * n_nodes
    diag_vals = np.array([4.0, 9.0, 16.0, 1.0, 25.0, 36.0])
    K_e = np.diag(diag_vals)
    K_g = -np.eye(size)
    boundary_conditions = {}
    (lam1, mode1) = fcn(K_e, K_g, boundary_conditions, n_nodes)
    c = 2.5
    (lam2, mode2) = fcn(K_e, c * K_g, boundary_conditions, n_nodes)
    assert np.isclose(lam2, lam1 / c, rtol=1e-08, atol=1e-12)
    assert mode1.shape == (size,)
    assert mode2.shape == (size,)
    assert np.linalg.norm(mode2) > 0.0
    idx1 = int(np.argmax(np.abs(mode1)))
    idx2 = int(np.argmax(np.abs(mode2)))
    assert idx1 == idx2