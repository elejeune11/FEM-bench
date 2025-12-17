def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 1
    diag_vals = np.array([5.0, 12.0, 2.5, 7.0, 9.0, 3.25], dtype=float)
    K_e = np.diag(diag_vals)
    K_g = -np.eye(6, dtype=float)
    boundary_conditions = {}
    lam, phi = fcn(K_e, K_g, boundary_conditions, n_nodes)
    assert np.isclose(lam, diag_vals.min(), rtol=1e-09, atol=1e-12)
    assert phi.shape == (6 * n_nodes,)
    assert np.linalg.norm(phi) > 0
    resid = K_e @ phi + lam * K_g @ phi
    assert np.allclose(resid, 0.0, atol=1e-08)
    min_idx = int(np.argmin(diag_vals))
    max_comp_idx = int(np.argmax(np.abs(phi)))
    assert max_comp_idx == min_idx

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned.
    """
    n_nodes = 1
    K_e = np.eye(6, dtype=float)
    K_g = np.zeros((6, 6), dtype=float)
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
    K_e[0, 1] = -1.0
    K_e[1, 0] = 1.0
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
    diag_vals = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)
    K_e = np.diag(diag_vals)
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
    diag_vals = np.array([3.0, 8.0, 2.5, 6.0, 12.0, 4.5], dtype=float)
    K_e = np.diag(diag_vals)
    K_g_base = -np.eye(6, dtype=float)
    boundary_conditions = {}
    lam_base, phi_base = fcn(K_e, K_g_base, boundary_conditions, n_nodes)
    assert phi_base.shape == (6 * n_nodes,)
    assert np.linalg.norm(phi_base) > 0
    resid_base = K_e @ phi_base + lam_base * K_g_base @ phi_base
    assert np.allclose(resid_base, 0.0, atol=1e-08)
    c = 2.75
    K_g_scaled = c * K_g_base
    lam_scaled, phi_scaled = fcn(K_e, K_g_scaled, boundary_conditions, n_nodes)
    assert phi_scaled.shape == (6 * n_nodes,)
    assert np.isclose(lam_scaled, lam_base / c, rtol=1e-09, atol=1e-12)
    resid_scaled = K_e @ phi_scaled + lam_scaled * K_g_scaled @ phi_scaled
    assert np.allclose(resid_scaled, 0.0, atol=1e-08)