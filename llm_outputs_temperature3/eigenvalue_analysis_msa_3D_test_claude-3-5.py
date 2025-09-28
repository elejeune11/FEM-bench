def test_eigen_known_answer(fcn):
    """Test eigenvalue analysis with simple diagonal matrices where solution is known."""
    n_nodes = 2
    K_e = np.diag([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    K_g = -np.eye(12)

    class BC:

        def __init__(self):
            self.constrained_dofs = [0]
    (lambda_cr, mode) = fcn(K_e, K_g, BC(), n_nodes)
    assert np.isclose(lambda_cr, 3.0)
    assert mode.shape == (12,)
    assert np.count_nonzero(mode) == 1

def test_eigen_singluar_detected(fcn):
    """Test that singular/ill-conditioned matrices raise ValueError."""
    n_nodes = 2
    K_e = np.zeros((12, 12))
    K_g = np.eye(12)

    class BC:

        def __init__(self):
            self.constrained_dofs = [0]
    with pytest.raises(ValueError, match='singular'):
        fcn(K_e, K_g, BC(), n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Test that complex eigenpairs raise ValueError."""
    n_nodes = 2
    K_e = np.array([[1, -1j], [1j, 1]])
    K_g = np.eye(2)

    class BC:

        def __init__(self):
            self.constrained_dofs = []
    with pytest.raises(ValueError, match='complex'):
        fcn(K_e, K_g, BC(), n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Test that absence of positive eigenvalues raises ValueError."""
    n_nodes = 2
    K_e = -np.eye(12)
    K_g = np.eye(12)

    class BC:

        def __init__(self):
            self.constrained_dofs = [0]
    with pytest.raises(ValueError, match='positive'):
        fcn(K_e, K_g, BC(), n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Test that eigenvalues scale inversely with K_g scaling."""
    n_nodes = 2
    K_e = np.diag([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    K_g = -np.eye(12)

    class BC:

        def __init__(self):
            self.constrained_dofs = [0]
    (lambda1, mode1) = fcn(K_e, K_g, BC(), n_nodes)
    (lambda2, mode2) = fcn(K_e, 2 * K_g, BC(), n_nodes)
    assert np.isclose(lambda1, 2 * lambda2)
    assert mode1.shape == mode2.shape == (12,)