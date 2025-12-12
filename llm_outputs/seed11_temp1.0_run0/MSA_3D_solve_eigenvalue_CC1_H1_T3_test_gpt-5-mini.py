def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 1
    size = 6 * n_nodes
    diag = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    K_e_global = np.diag(diag)
    K_g_global = -np.eye(size)
    boundary_conditions = {}
    (lam, mode) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isfinite(lam)
    assert lam == pytest.approx(diag.min(), rel=1e-08, abs=1e-12)
    assert mode.shape == (size,)
    idx_min = int(np.argmin(diag))
    tol = 1e-08
    for i in range(size):
        if i == idx_min:
            assert abs(mode[i]) > 1e-12
        else:
            assert abs(mode[i]) < tol

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 1
    size = 6 * n_nodes
    K_e_global = np.zeros((size, size))
    K_g_global = np.eye(size)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 1
    size = 6 * n_nodes
    bc = np.array([False, False, True, True, True, True], dtype=bool)
    boundary_conditions = {0: bc}
    free_mask = _make_free_mask_from_bc(boundary_conditions, n_nodes)
    K_e_ff = np.eye(2)
    K_g_ff = np.array([[0.0, 1.0], [-1.0, 0.0]])
    K_e_global = _assemble_full_matrix(n_nodes, K_e_ff, free_mask)
    K_g_global = _assemble_full_matrix(n_nodes, K_g_ff, free_mask)
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 1
    size = 6 * n_nodes
    K_e_global = np.eye(size)
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
    size = 6 * n_nodes
    diag = np.array([8.0, 12.0, 16.0, 20.0, 24.0, 28.0])
    K_e_global = np.diag(diag)
    K_g_global = -np.eye(size)
    c = 3.7
    K_g_scaled = c * K_g_global
    bc = {}
    (lam1, mode1) = fcn(K_e_global, K_g_global, bc, n_nodes)
    (lam2, mode2) = fcn(K_e_global, K_g_scaled, bc, n_nodes)
    assert lam1 == pytest.approx(lam2 * c, rel=1e-08, abs=1e-12) or lam2 == pytest.approx(lam1 / c, rel=1e-08, abs=1e-12)
    assert mode1.shape == (size,)
    assert mode2.shape == (size,)
    idx1 = np.argmax(np.abs(mode1))
    idx2 = np.argmax(np.abs(mode2))
    assert idx1 == idx2