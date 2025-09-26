def test_eigen_known_answer(fcn):
    """Verify eigenvalue analysis with diagonal matrices having known solution."""
    n_nodes = 2
    n_dof = 6 * n_nodes
    K_e = np.diag([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0])
    K_g = -np.eye(n_dof)
    bc = {'fixed_dofs': [0, 1, 2, 3, 4, 5]}
    (lambda_cr, mode) = fcn(K_e, K_g, bc, n_nodes)
    assert np.isclose(lambda_cr, 14.0)
    expected_mode = np.zeros(n_dof)
    expected_mode[6] = 1.0
    assert np.allclose(mode / np.max(np.abs(mode)), expected_mode / np.max(np.abs(expected_mode)))

def test_eigen_singluar_detected(fcn):
    """Verify detection of singular/ill-conditioned matrices."""
    n_nodes = 2
    n_dof = 6 * n_nodes
    K_e = np.ones((n_dof, n_dof)) * 1e-20
    K_g = -np.eye(n_dof)
    bc = {'fixed_dofs': [0, 1, 2, 3, 4, 5]}
    with pytest.raises(ValueError, match='singular|ill-conditioned'):
        fcn(K_e, K_g, bc, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify detection of complex eigenpairs."""
    n_nodes = 2
    n_dof = 6 * n_nodes
    K_e = np.eye(n_dof)
    K_g = 1j * np.eye(n_dof)
    bc = {'fixed_dofs': [0, 1, 2, 3, 4, 5]}
    with pytest.raises(ValueError, match='complex'):
        fcn(K_e, K_g, bc, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify detection when no positive eigenvalues exist."""
    n_nodes = 2
    n_dof = 6 * n_nodes
    K_e = -np.eye(n_dof)
    K_g = np.eye(n_dof)
    bc = {'fixed_dofs': [0, 1, 2, 3, 4, 5]}
    with pytest.raises(ValueError, match='positive'):
        fcn(K_e, K_g, bc, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Verify correct scaling behavior with reference load."""
    n_nodes = 2
    n_dof = 6 * n_nodes
    K_e = np.diag(np.linspace(1, n_dof, n_dof))
    K_g = -np.eye(n_dof)
    bc = {'fixed_dofs': [0, 1, 2, 3, 4, 5]}
    (lambda1, mode1) = fcn(K_e, K_g, bc, n_nodes)
    scale = 2.0
    (lambda2, mode2) = fcn(K_e, scale * K_g, bc, n_nodes)
    assert np.isclose(lambda1, scale * lambda2)
    assert np.allclose(mode1 / np.max(np.abs(mode1)), mode2 / np.max(np.abs(mode2)))