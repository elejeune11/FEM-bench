def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    ndof = 6 * n_nodes
    K_e = np.diag(np.arange(1, ndof + 1, dtype=float) * 10.0)
    K_g = -np.eye(ndof)
    boundary_conditions = {}
    (lam, mode) = fcn(K_e, K_g, boundary_conditions, n_nodes)
    assert np.isclose(lam, 10.0)
    assert abs(mode[0]) > 1e-05
    assert np.all(np.abs(mode[1:]) < 1e-05 * np.abs(mode[0]))

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 1
    ndof = 6
    K_e = np.eye(ndof)
    K_e[2, 2] = 0.0
    K_g = -np.eye(ndof)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 1
    ndof = 6
    K_e = np.eye(ndof)
    K_e[0, 1] = 10.0
    K_e[1, 0] = -10.0
    K_g = -np.eye(ndof)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 1
    ndof = 6
    K_e = np.eye(ndof)
    K_g = np.eye(ndof)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 2
    ndof = 6 * n_nodes
    K_e = np.eye(ndof) * 100.0
    K_g_base = -np.eye(ndof)
    boundary_conditions = {}
    (lam_base, mode_base) = fcn(K_e, K_g_base, boundary_conditions, n_nodes)
    scaling_factor = 2.0
    K_g_scaled = K_g_base * scaling_factor
    (lam_scaled, mode_scaled) = fcn(K_e, K_g_scaled, boundary_conditions, n_nodes)
    assert np.isclose(lam_scaled, lam_base / scaling_factor)
    assert mode_scaled.shape == (ndof,)