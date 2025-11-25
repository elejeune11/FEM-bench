def test_eigen_known_answer(fcn):
    """Test eigenvalue analysis with simple diagonal matrices where solution is known."""
    n = 2
    K_e = np.diag([2.0, 4.0, 6.0, 8.0, 10.0, 12.0] * n)
    K_g = -np.eye(6 * n)
    bc = {'fixed_dofs': [0, 1, 2, 3, 4, 5]}
    (lambda_cr, mode) = fcn(K_e, K_g, bc, n)
    assert np.isclose(lambda_cr, 2.0)
    assert mode.shape == (12,)
    assert np.allclose(mode[:6], 0)

def test_eigen_singluar_detected(fcn):
    """Test that singular/ill-conditioned matrices raise ValueError."""
    n = 2
    K_e = np.zeros((6 * n, 6 * n))
    K_g = np.eye(6 * n)
    bc = {'fixed_dofs': [0, 1, 2]}
    with pytest.raises(ValueError, match='singular'):
        fcn(K_e, K_g, bc, n)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Test that complex eigenpairs raise ValueError."""
    n = 2
    K_e = np.array([[1, -1j], [1j, 1]])
    K_g = np.eye(2)
    bc = {'fixed_dofs': []}
    with pytest.raises(ValueError, match='complex'):
        fcn(K_e, K_g, bc, n)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Test that absence of positive eigenvalues raises ValueError."""
    n = 2
    K_e = -np.eye(6 * n)
    K_g = np.eye(6 * n)
    bc = {'fixed_dofs': [0, 1, 2, 3, 4, 5]}
    with pytest.raises(ValueError, match='positive'):
        fcn(K_e, K_g, bc, n)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Test that eigenvalues scale inversely with K_g scaling."""
    n = 2
    K_e = np.diag([2.0, 4.0, 6.0, 8.0, 10.0, 12.0] * n)
    K_g = -np.eye(6 * n)
    bc = {'fixed_dofs': [0, 1, 2, 3, 4, 5]}
    (lambda1, mode1) = fcn(K_e, K_g, bc, n)
    scale = 2.0
    (lambda2, mode2) = fcn(K_e, scale * K_g, bc, n)
    assert np.isclose(lambda1 / scale, lambda2)
    assert mode1.shape == mode2.shape == (12,)