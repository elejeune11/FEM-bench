def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. With diagonal K_e and K_g = -I, the critical load
    factors reduce to the diagonal entries of K_e, so the function should return
    the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 2
    ndof = 6 * n_nodes
    d = np.array([10.0, 7.0, 5.0, 9.0, 2.0, 8.0, 11.0, 12.0, 6.0, 4.0, 13.0, 14.0])
    assert d.size == ndof
    K_e = np.diag(d)
    K_g = -np.eye(ndof)
    boundary_conditions = {}
    lam, v = fcn(K_e, K_g, boundary_conditions, n_nodes)
    expected_idx = int(np.argmin(d))
    expected_lambda = float(d[expected_idx])
    assert np.isclose(lam, expected_lambda, rtol=1e-12, atol=1e-12)
    assert v.shape == (ndof,)
    assert np.linalg.norm(v) > 0
    max_idx = int(np.argmax(np.abs(v)))
    assert max_idx == expected_idx
    max_val = np.max(np.abs(v)) if np.max(np.abs(v)) != 0 else 1.0
    mask = np.ones(ndof, dtype=bool)
    mask[expected_idx] = False
    assert np.all(np.abs(v[mask]) <= 1e-10 * max_val)
    residual = K_e @ v + lam * K_g @ v
    assert np.linalg.norm(residual) <= 1e-10 * (np.linalg.norm(K_e @ v) + abs(lam) * np.linalg.norm(K_g @ v) + 1e-30)

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the reduced elastic block
    is singular/ill-conditioned.
    """
    n_nodes = 1
    ndof = 6 * n_nodes
    diag_vals = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    K_e = np.diag(diag_vals)
    K_g = -np.eye(ndof)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized eigenproblem
    yields significantly complex pairs.
    """
    n_nodes = 1
    ndof = 6 * n_nodes
    R = np.array([[0.0, -1.0], [1.0, 0.0]])
    rest = np.diag([2.0, 3.0, 4.0, 5.0])
    K_e = np.block([[R, np.zeros((2, 4))], [np.zeros((4, 2)), rest]])
    K_g = -np.eye(ndof)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive eigenvalues are present.
    """
    n_nodes = 1
    ndof = 6 * n_nodes
    d = np.array([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    K_e = np.diag(d)
    K_g = -np.eye(ndof)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """
    Check that the computed critical load factor scales correctly with the reference
    geometric stiffness. Scaling K_g by a constant c should scale the reported eigenvalue
    by 1/c, while still returning valid global mode vectors of the correct size.
    """
    n_nodes = 2
    ndof = 6 * n_nodes
    d = np.array([4.0, 10.0, 6.0, 8.0, 3.0, 7.0, 11.0, 9.0, 12.0, 5.0, 13.0, 14.0])
    K_e = np.diag(d)
    K_g_base = -np.eye(ndof)
    boundary_conditions = {}
    lam1, v1 = fcn(K_e, K_g_base, boundary_conditions, n_nodes)
    c = 2.5
    K_g_scaled = c * K_g_base
    lam2, v2 = fcn(K_e, K_g_scaled, boundary_conditions, n_nodes)
    assert v1.shape == (ndof,)
    assert v2.shape == (ndof,)
    assert np.linalg.norm(v1) > 0
    assert np.linalg.norm(v2) > 0
    assert np.isclose(lam2, lam1 / c, rtol=1e-12, atol=1e-12)
    idx_expected = int(np.argmin(d))
    idx1 = int(np.argmax(np.abs(v1)))
    idx2 = int(np.argmax(np.abs(v2)))
    assert idx1 == idx_expected
    assert idx2 == idx_expected
    res1 = K_e @ v1 + lam1 * K_g_base @ v1
    res2 = K_e @ v2 + lam2 * K_g_scaled @ v2
    tol1 = 1e-10 * (np.linalg.norm(K_e @ v1) + abs(lam1) * np.linalg.norm(K_g_base @ v1) + 1e-30)
    tol2 = 1e-10 * (np.linalg.norm(K_e @ v2) + abs(lam2) * np.linalg.norm(K_g_scaled @ v2) + 1e-30)
    assert np.linalg.norm(res1) <= tol1
    assert np.linalg.norm(res2) <= tol2