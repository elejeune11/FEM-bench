def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 1
    K_e = np.zeros((6, 6))
    K_e[0, 0] = 100.0
    K_e[1, 1] = 200.0
    for i in range(2, 6):
        K_e[i, i] = 1000000000.0
    K_g = np.zeros((6, 6))
    K_g[0, 0] = -1.0
    K_g[1, 1] = -1.0
    boundary_conditions = {0: [False, False, True, True, True, True]}
    (eig_val, eig_vec) = fcn(K_e, K_g, boundary_conditions, n_nodes)
    assert np.isclose(eig_val, 100.0)
    assert eig_vec.shape == (6,)
    assert np.allclose(eig_vec[2:], 0.0)
    assert not np.isclose(eig_vec[0], 0.0)
    assert np.isclose(eig_vec[1], 0.0)

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned.
    """
    n_nodes = 1
    boundary_conditions = {0: [False, True, True, True, True, True]}
    K_e = np.zeros((6, 6))
    K_g = np.zeros((6, 6))
    K_g[0, 0] = -1.0
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """
    n_nodes = 1
    boundary_conditions = {0: [False, False, True, True, True, True]}
    K_e = np.eye(6)
    K_e[0, 1] = -10.0
    K_e[1, 0] = 10.0
    K_g = np.zeros((6, 6))
    K_g[0, 0] = -1.0
    K_g[1, 1] = -1.0
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 1
    boundary_conditions = {0: [False, True, True, True, True, True]}
    K_e = np.eye(6) * 10.0
    K_g = np.eye(6) * 1.0
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
    boundary_conditions = {0: [False, True, True, True, True, True]}
    K_e = np.eye(6) * 100.0
    K_g_base = np.zeros((6, 6))
    K_g_base[0, 0] = -1.0
    (lam_1, phi_1) = fcn(K_e, K_g_base, boundary_conditions, n_nodes)
    scaling_factor = 2.0
    K_g_scaled = K_g_base * scaling_factor
    (lam_2, phi_2) = fcn(K_e, K_g_scaled, boundary_conditions, n_nodes)
    assert np.isclose(lam_2, lam_1 / scaling_factor)
    assert phi_2.shape == (6,)
    assert not np.allclose(phi_2, 0.0)