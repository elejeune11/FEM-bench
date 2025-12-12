def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 1
    d = np.array([3.0, 5.0, 2.0, 7.0, 11.0, 1.5], dtype=float)
    K_e = np.diag(d)
    K_g = -np.eye(6, dtype=float)
    boundary_conditions = {}
    (lam, mode) = fcn(K_e, K_g, boundary_conditions, n_nodes)
    assert np.isclose(lam, d.min(), rtol=0, atol=1e-12)
    assert mode.shape == (6,)
    idx_min = int(np.argmin(d))
    assert int(np.argmax(np.abs(mode))) == idx_min
    assert np.allclose(np.delete(mode, idx_min), 0.0, atol=1e-10)

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned.
    """
    n_nodes = 1
    K_e = np.diag([1.0, 0.0, 2.0, 3.0, 4.0, 5.0])
    K_g = -np.eye(6, dtype=float)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """
    n_nodes = 1
    K_e = np.zeros((6, 6), dtype=float)
    K_e[:2, :2] = np.array([[0.0, -1.0], [1.0, 0.0]])
    K_e[2, 2] = 2.0
    K_e[3, 3] = 3.0
    K_e[4, 4] = 4.0
    K_e[5, 5] = 5.0
    K_g = -np.eye(6, dtype=float)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 1
    K_e = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    K_g = np.eye(6, dtype=float)
    boundary_conditions = {}
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
    d = np.array([2.0, 4.0, 1.0, 3.0, 5.0, 6.0], dtype=float)
    K_e = np.diag(d)
    K_g = -np.eye(6, dtype=float)
    boundary_conditions = {}
    (lam1, mode1) = fcn(K_e, K_g, boundary_conditions, n_nodes)
    c = 7.5
    (lam2, mode2) = fcn(K_e, c * K_g, boundary_conditions, n_nodes)
    assert np.isclose(lam2, lam1 / c, rtol=0, atol=1e-12)
    assert mode1.shape == (6,)
    assert mode2.shape == (6,)
    idx_min = int(np.argmin(d))
    assert int(np.argmax(np.abs(mode1))) == idx_min
    assert int(np.argmax(np.abs(mode2))) == idx_min