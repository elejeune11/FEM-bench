def test_eigen_known_answer(fcn):
    """Verify eigenvalue analysis with diagonal matrices and known solution."""
    n_nodes = 2
    K_e = np.diag([2.0, 4.0, 6.0, 8.0, 10.0, 12.0] * n_nodes)
    K_g = -np.eye(6 * n_nodes)
    bc = {'fixed_dofs': [0, 1, 2, 3, 4, 5]}
    (lambda_cr, mode) = fcn(K_e, K_g, bc, n_nodes)
    assert np.isclose(lambda_cr, 8.0)
    assert mode.shape == (12,)
    assert np.all(mode[:6] == 0)

def test_eigen_singluar_detected(fcn):
    """Verify detection of singular/ill-conditioned matrices."""
    n_nodes = 2
    K_e = np.zeros((12, 12))
    K_g = -np.eye(12)
    bc = {'fixed_dofs': [0, 1, 2]}
    with pytest.raises(ValueError, match='singular'):
        fcn(K_e, K_g, bc, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify detection of complex eigenpairs."""
    n_nodes = 2
    K_e = np.array([[1, -1j], [1j, 1]])
    K_g = -np.eye(2)
    bc = {'fixed_dofs': []}
    with pytest.raises(ValueError, match='complex'):
        fcn(K_e, K_g, bc, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify detection when no positive eigenvalues exist."""
    n_nodes = 2
    K_e = -np.eye(12)
    K_g = np.eye(12)
    bc = {'fixed_dofs': [0, 1, 2, 3, 4, 5]}
    with pytest.raises(ValueError, match='positive'):
        fcn(K_e, K_g, bc, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Verify correct scaling behavior with reference load changes."""
    n_nodes = 2
    K_e = np.diag([2.0, 4.0, 6.0, 8.0, 10.0, 12.0] * n_nodes)
    K_g = -np.eye(6 * n_nodes)
    bc = {'fixed_dofs': [0, 1, 2, 3, 4, 5]}
    (lambda1, mode1) = fcn(K_e, K_g, bc, n_nodes)
    scale = 2.0
    (lambda2, mode2) = fcn(K_e, scale * K_g, bc, n_nodes)
    assert np.isclose(lambda1, scale * lambda2)
    assert mode1.shape == mode2.shape == (12,)
    assert np.allclose(np.abs(mode1), np.abs(mode2))