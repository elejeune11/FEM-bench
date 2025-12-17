def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. With diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e. The function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 1
    K_e_global = np.diag([3.0, 1.0, 5.0, 2.0, 4.0, 7.0])
    K_g_global = -np.eye(6)
    boundary_conditions = {}
    lam, vec = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert lam == pytest.approx(1.0, rel=1e-12, abs=1e-12)
    assert vec.shape == (6,)
    assert np.allclose(K_e_global @ vec + lam * K_g_global @ vec, 0.0, rtol=1e-10, atol=1e-12)
    expected_index = 1
    assert int(np.argmax(np.abs(vec))) == expected_index
    others = np.copy(vec)
    others[expected_index] = 0.0
    assert np.linalg.norm(others) <= 1e-08 * max(np.abs(vec[expected_index]), 1e-12)

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned.
    """
    n_nodes = 1
    K_e_global = np.zeros((6, 6))
    K_e_global[0, 0] = 1.0
    K_e_global[1, 1] = 1e-30
    K_g_global = -np.eye(6)
    boundary_conditions = {0: [False, False, True, True, True, True]}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """
    n_nodes = 1
    K_e_global = np.zeros((6, 6))
    K_e_global[0, 1] = -1.0
    K_e_global[1, 0] = 1.0
    K_g_global = -np.eye(6)
    boundary_conditions = {0: [False, False, True, True, True, True]}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 1
    K_e_global = np.zeros((6, 6))
    K_e_global[0, 0] = -1.0
    K_e_global[1, 1] = -2.0
    K_g_global = -np.eye(6)
    boundary_conditions = {0: [False, False, True, True, True, True]}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """
    Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size.
    """
    n_nodes = 1
    K_e_global = np.diag([2.0, 0.5, 1.5, 5.0, 3.0, 4.5])
    K_g_global = -np.eye(6)
    boundary_conditions = {0: [False, False, False, True, False, True]}
    lam1, vec1 = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert vec1.shape == (6,)
    assert vec1[3] == pytest.approx(0.0, abs=1e-14)
    assert vec1[5] == pytest.approx(0.0, abs=1e-14)
    c = 3.7
    lam2, vec2 = fcn(K_e_global, c * K_g_global, boundary_conditions, n_nodes)
    assert vec2.shape == (6,)
    assert vec2[3] == pytest.approx(0.0, abs=1e-14)
    assert vec2[5] == pytest.approx(0.0, abs=1e-14)
    assert lam2 == pytest.approx(lam1 / c, rel=1e-10, abs=1e-12)