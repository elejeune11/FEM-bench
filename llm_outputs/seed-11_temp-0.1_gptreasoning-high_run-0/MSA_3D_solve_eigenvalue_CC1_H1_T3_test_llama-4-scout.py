def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case.
    """
    n_nodes = 1
    K_e_global = np.eye(6)
    K_g_global = -np.eye(6)
    boundary_conditions = {}
    (eigenvalue, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(eigenvalue, 1.0)
    assert np.allclose(mode_shape, np.array([1, 0, 0, 0, 0, 0]))

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned.
    """
    n_nodes = 1
    K_e_global = np.eye(6)
    K_e_global[0, 0] = 0
    K_g_global = -np.eye(6)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """
    n_nodes = 1
    K_e_global = np.eye(6)
    K_g_global = np.array([[1, 1j, 0, 0, 0, 0], [-1j, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 1
    K_e_global = np.eye(6)
    K_g_global = np.eye(6)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """
    Check that the computed critical load factor scales correctly with the
    reference geometric stiffness.
    """
    n_nodes = 1
    K_e_global = np.eye(6)
    K_g_global = -np.eye(6)
    boundary_conditions = {}
    (eigenvalue1, mode_shape1) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    (eigenvalue2, mode_shape2) = fcn(K_e_global, 2 * K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(eigenvalue2, 0.5 * eigenvalue1)
    assert np.allclose(mode_shape1, mode_shape2)