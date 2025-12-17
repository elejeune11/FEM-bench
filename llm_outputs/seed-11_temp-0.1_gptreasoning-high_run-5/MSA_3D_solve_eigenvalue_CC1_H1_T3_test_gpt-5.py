def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. With diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e. The function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 1
    Ke_diag = np.array([3.0, 5.0, 7.0, 11.0, 13.0, 17.0])
    K_e = np.diag(Ke_diag)
    K_g = -np.eye(6 * n_nodes)
    boundary_conditions = {}
    lam, mode = fcn(K_e, K_g, boundary_conditions, n_nodes)
    assert np.isclose(lam, Ke_diag.min(), rtol=1e-12, atol=1e-12)
    assert mode.shape == (6 * n_nodes,)
    peak_idx = int(np.argmax(np.abs(mode)))
    assert peak_idx == 0
    off_peak_norm = np.linalg.norm(np.delete(mode, peak_idx))
    denom = abs(mode[peak_idx]) if abs(mode[peak_idx]) > 0 else 1.0
    assert off_peak_norm / denom < 1e-08

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic/geometric block is singular/ill-conditioned.
    """
    n_nodes = 1
    K_e = np.eye(6 * n_nodes)
    K_g = np.zeros((6 * n_nodes, 6 * n_nodes))
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """
    n_nodes = 1
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    K_e = np.kron(np.eye(3), J)
    K_g = -np.eye(6 * n_nodes)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 1
    Ke_diag = np.array([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    K_e = np.diag(Ke_diag)
    K_g = -np.eye(6 * n_nodes)
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
    n_nodes = 2
    Ke_diag = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 3.0, 70.0, 80.0, 90.0, 100.0, 110.0])
    K_e = np.diag(Ke_diag)
    K_g = -np.eye(6 * n_nodes)
    boundary_conditions = {0: np.array([True, False, False, True, False, True], dtype=bool)}
    lam1, mode1 = fcn(K_e, K_g, boundary_conditions, n_nodes)
    c = 7.0
    lam2, mode2 = fcn(K_e, c * K_g, boundary_conditions, n_nodes)
    assert np.isclose(lam2, lam1 / c, rtol=1e-12, atol=1e-12)
    assert mode1.shape == (6 * n_nodes,)
    assert mode2.shape == (6 * n_nodes,)
    fixed_mask = np.zeros(6 * n_nodes, dtype=bool)
    for node, mask in boundary_conditions.items():
        start = node * 6
        fixed_mask[start:start + 6] = mask
    assert np.allclose(mode1[fixed_mask], 0.0)
    assert np.allclose(mode2[fixed_mask], 0.0)