def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    np = fcn.__globals__.get('np') or fcn.__globals__.get('numpy')
    n_nodes = 1
    ndof = 6 * n_nodes
    diag_vals = np.array([5.0, 3.0, 7.0, 10.0, 2.0, 6.0], dtype=float)
    K_e = np.diag(diag_vals)
    K_g = -np.eye(ndof, dtype=float)
    free = np.arange(ndof, dtype=int)
    constrained = np.array([], dtype=int)
    bc = (free, constrained)
    (lam, mode) = fcn(K_e, K_g, bc, n_nodes)
    assert np.isfinite(lam)
    assert np.isclose(lam, 2.0, rtol=1e-12, atol=1e-14)
    assert mode.shape == (ndof,)
    idx = int(np.argmax(np.abs(mode)))
    assert idx == 4
    assert np.abs(mode[4]) > 1e-12
    mask = np.ones(ndof, dtype=bool)
    mask[4] = False
    assert np.all(np.abs(mode[mask]) <= 1e-08)

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    np = fcn.__globals__.get('np') or fcn.__globals__.get('numpy')
    n_nodes = 1
    ndof = 6 * n_nodes
    diag_vals = np.array([1.0, 1e-18, 2.0, 3.0, 4.0, 5.0], dtype=float)
    K_e = np.diag(diag_vals)
    K_g = -np.eye(ndof, dtype=float)
    free = np.arange(ndof, dtype=int)
    constrained = np.array([], dtype=int)
    bc = (free, constrained)
    raised = False
    try:
        fcn(K_e, K_g, bc, n_nodes)
    except ValueError:
        raised = True
    assert raised

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    np = fcn.__globals__.get('np') or fcn.__globals__.get('numpy')
    n_nodes = 1
    ndof = 6 * n_nodes
    K_e = np.eye(ndof, dtype=float)
    K_g = -np.eye(ndof, dtype=float)
    K_g[:2, :2] = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float)
    free = np.arange(ndof, dtype=int)
    constrained = np.array([], dtype=int)
    bc = (free, constrained)
    raised = False
    try:
        fcn(K_e, K_g, bc, n_nodes)
    except ValueError:
        raised = True
    assert raised

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    np = fcn.__globals__.get('np') or fcn.__globals__.get('numpy')
    n_nodes = 1
    ndof = 6 * n_nodes
    diag_vals = -np.array([1.0, 3.0, 2.0, 4.0, 6.0, 5.0], dtype=float)
    K_e = np.diag(diag_vals)
    K_g = -np.eye(ndof, dtype=float)
    free = np.arange(ndof, dtype=int)
    constrained = np.array([], dtype=int)
    bc = (free, constrained)
    raised = False
    try:
        fcn(K_e, K_g, bc, n_nodes)
    except ValueError:
        raised = True
    assert raised

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    np = fcn.__globals__.get('np') or fcn.__globals__.get('numpy')
    n_nodes = 1
    ndof = 6 * n_nodes
    diag_vals = np.array([5.0, 3.0, 7.0, 10.0, 2.0, 6.0], dtype=float)
    K_e = np.diag(diag_vals)
    K_g = -np.eye(ndof, dtype=float)
    c = 4.3
    free = np.arange(ndof, dtype=int)
    constrained = np.array([], dtype=int)
    bc = (free, constrained)
    (lam1, mode1) = fcn(K_e, K_g, bc, n_nodes)
    (lam2, mode2) = fcn(K_e, c * K_g, bc, n_nodes)
    assert np.isclose(lam2, lam1 / c, rtol=1e-12, atol=1e-14)
    assert mode1.shape == (ndof,)
    assert mode2.shape == (ndof,)
    assert np.linalg.norm(mode1) > 0.0
    assert np.linalg.norm(mode2) > 0.0