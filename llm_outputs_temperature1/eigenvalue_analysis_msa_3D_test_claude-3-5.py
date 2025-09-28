def test_eigen_known_answer(fcn):
    """Test eigenvalue analysis with a simple diagonal system having known solution."""
    n_nodes = 2
    K_e = np.diag([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    K_g = -np.eye(12)

    class BC:

        def __init__(self):
            self.constrained_dofs = []
    bc = BC()
    (factor, mode) = fcn(K_e, K_g, bc, n_nodes)
    assert np.isclose(factor, 2.0)
    assert np.argmax(np.abs(mode)) == 0

def test_eigen_singluar_detected(fcn):
    """Test that singular/ill-conditioned matrix is detected."""
    n_nodes = 2
    K_e = np.zeros((12, 12))
    K_g = -np.eye(12)

    class BC:

        def __init__(self):
            self.constrained_dofs = []
    bc = BC()
    with pytest.raises(ValueError, match='singular'):
        fcn(K_e, K_g, bc, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Test detection of complex eigenpairs."""
    n_nodes = 2
    K_e = np.array([[1, -1j], [1j, 1]])
    K_g = -np.eye(2)

    class BC:

        def __init__(self):
            self.constrained_dofs = []
    bc = BC()
    with pytest.raises(ValueError, match='complex'):
        fcn(K_e, K_g, bc, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Test detection of case with no positive eigenvalues."""
    n_nodes = 2
    K_e = -np.eye(12)
    K_g = np.eye(12)

    class BC:

        def __init__(self):
            self.constrained_dofs = []
    bc = BC()
    with pytest.raises(ValueError, match='positive'):
        fcn(K_e, K_g, bc, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Test that eigenvalues scale inversely with K_g scaling."""
    n_nodes = 2
    K_e = np.diag([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    K_g = -np.eye(12)

    class BC:

        def __init__(self):
            self.constrained_dofs = []
    bc = BC()
    (factor1, mode1) = fcn(K_e, K_g, bc, n_nodes)
    (factor2, mode2) = fcn(K_e, 2 * K_g, bc, n_nodes)
    assert np.isclose(factor1, 2 * factor2)
    assert np.allclose(mode1, mode2)