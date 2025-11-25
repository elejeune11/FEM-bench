def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case with diagonal K_e and K_g = -I. The critical load
    factor should be the smallest diagonal entry of K_e, and the mode should align
    with the corresponding DOF."""
    n_nodes = 1
    ndof = 6 * n_nodes
    K_e = np.diag([5.0, 3.0, 7.0, 2.0, 9.0, 4.0])
    K_g = -np.eye(ndof)
    free = np.arange(ndof, dtype=int)
    constrained = np.array([], dtype=int)
    mod = importlib.import_module(fcn.__module__)

    def _part(bc, n):
        return (free, constrained)
    setattr(mod, 'partition_degrees_of_freedom', _part)
    (lam, vec) = fcn(K_e, K_g, boundary_conditions=object(), n_nodes=n_nodes)
    assert np.isclose(lam, 2.0, rtol=1e-12, atol=1e-12)
    max_idx = int(np.argmax(np.abs(vec)))
    assert max_idx == 3
    dominant = np.abs(vec[max_idx])
    assert dominant > 0
    off_diag_norm = np.linalg.norm(np.delete(vec, max_idx))
    assert off_diag_norm <= 1e-10 * dominant

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the reduced elastic
    block is singular/ill-conditioned."""
    n_nodes = 1
    ndof = 6 * n_nodes
    K_e = np.zeros((ndof, ndof), dtype=float)
    K_g = -np.eye(ndof)
    free = np.arange(ndof, dtype=int)
    constrained = np.array([], dtype=int)
    mod = importlib.import_module(fcn.__module__)

    def _part(bc, n):
        return (free, constrained)
    setattr(mod, 'partition_degrees_of_freedom', _part)
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions=object(), n_nodes=n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 1
    ndof = 6 * n_nodes
    K_e = np.eye(ndof)
    K_g = np.zeros((ndof, ndof), dtype=float)
    K_g[0:2, 0:2] = np.array([[0.0, 1.0], [-1.0, 0.0]])
    free = np.array([0, 1], dtype=int)
    constrained = np.array([2, 3, 4, 5], dtype=int)
    mod = importlib.import_module(fcn.__module__)

    def _part(bc, n):
        return (free, constrained)
    setattr(mod, 'partition_degrees_of_freedom', _part)
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions=object(), n_nodes=n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive eigenvalues are present."""
    n_nodes = 1
    ndof = 6 * n_nodes
    K_e = -np.eye(ndof)
    K_g = -np.eye(ndof)
    free = np.arange(ndof, dtype=int)
    constrained = np.array([], dtype=int)
    mod = importlib.import_module(fcn.__module__)

    def _part(bc, n):
        return (free, constrained)
    setattr(mod, 'partition_degrees_of_freedom', _part)
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions=object(), n_nodes=n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the reference geometric stiffness.
    Scaling K_g by a constant c should scale the reported eigenvalue by 1/c, while still returning valid
    global mode vectors of the correct size."""
    n_nodes = 1
    ndof = 6 * n_nodes
    K_e = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    K_g = -np.eye(ndof)
    free = np.arange(ndof, dtype=int)
    constrained = np.array([], dtype=int)
    mod = importlib.import_module(fcn.__module__)

    def _part(bc, n):
        return (free, constrained)
    setattr(mod, 'partition_degrees_of_freedom', _part)
    (lam1, vec1) = fcn(K_e, K_g, boundary_conditions=object(), n_nodes=n_nodes)
    c = 5.0
    (lam2, vec2) = fcn(K_e, c * K_g, boundary_conditions=object(), n_nodes=n_nodes)
    assert np.isclose(lam2, lam1 / c, rtol=1e-10, atol=1e-12)
    assert vec1.shape == (ndof,)
    assert vec2.shape == (ndof,)
    assert np.all(np.isfinite(vec1))
    assert np.all(np.isfinite(vec2))
    assert np.linalg.norm(vec1) > 0
    assert np.linalg.norm(vec2) > 0