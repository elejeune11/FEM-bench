def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 1
    size = 6
    K_e_global = np.diag([3.0, 1.0, 10.0, 6.0, 100.0, 4.0])
    K_g_global = -np.eye(size)
    boundary_conditions = {}
    lam, mode = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isfinite(lam)
    assert lam > 0
    assert abs(lam - 1.0) < 1e-10
    assert mode.shape == (size,)
    assert int(np.argmax(np.abs(mode))) == 1

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 1
    size = 6
    K_e_global = np.diag([1.0, 2.0, 0.0, 3.0, 4.0, 5.0])
    K_g_global = -np.eye(size)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 1
    size = 6
    K_e_global = np.zeros((size, size))
    K_e_global[0:2, 0:2] = np.array([[0.0, 1.0], [-1.0, 0.0]])
    K_g_global = np.zeros((size, size))
    K_g_global[0:2, 0:2] = -np.eye(2)
    boundary_conditions = {0: np.array([False, False, True, True, True, True], dtype=bool)}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 1
    size = 6
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    K_g_global = np.eye(size)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 1
    size = 6
    K_e_global = np.diag([2.0, 5.0, 3.0, 7.0, 11.0, 13.0])
    K_g_global = -np.eye(size)
    boundary_conditions = {}
    lam1, mode1 = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    c = 2.5
    lam2, mode2 = fcn(K_e_global, c * K_g_global, boundary_conditions, n_nodes)
    assert mode1.shape == (size,)
    assert mode2.shape == (size,)
    assert np.isclose(lam2, lam1 / c, rtol=1e-10, atol=1e-12)
    idx1 = int(np.argmax(np.abs(mode1)))
    idx2 = int(np.argmax(np.abs(mode2)))
    assert idx1 == idx2 == 0