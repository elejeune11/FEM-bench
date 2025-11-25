def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. With diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e; the function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 1
    ndof = 6 * n_nodes
    K_e = np.diag([2.0, 5.0, 1.5, 7.0, 3.0, 4.0])
    K_g = -np.eye(ndof)
    boundary_conditions = {'constrained_dofs': []}
    (lam, mode) = fcn(K_e, K_g, boundary_conditions, n_nodes)
    assert mode.shape == (ndof,)
    assert abs(lam - 1.5) < 1e-10
    idx_max = np.argmax(np.abs(mode))
    assert idx_max == 2
    other_indices = [i for i in range(ndof) if i != idx_max]
    if np.abs(mode[idx_max]) > 0:
        assert np.max(np.abs(mode[other_indices])) <= 1e-06 * np.abs(mode[idx_max])

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the reduced elastic
    block is singular/ill-conditioned beyond the tolerance.
    """
    n_nodes = 1
    ndof = 6 * n_nodes
    diag_values = np.array([1e-20, 1.0, 2.0, 3.0, 4.0, 5.0])
    K_e = np.diag(diag_values)
    K_g = -np.eye(ndof)
    boundary_conditions = {'constrained_dofs': []}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """
    n_nodes = 1
    ndof = 6 * n_nodes
    K_e = np.eye(ndof)
    K_g = -np.eye(ndof)
    K_g[0, 0] = 0.0
    K_g[1, 1] = 0.0
    K_g[0, 1] = 1.0
    K_g[1, 0] = -1.0
    boundary_conditions = {'constrained_dofs': []}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 1
    ndof = 6 * n_nodes
    K_e = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    K_g = -np.eye(ndof)
    boundary_conditions = {'constrained_dofs': []}
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
    ndof = 6 * n_nodes
    K_e = np.diag([2.0, 5.0, 1.5, 7.0, 3.0, 4.0])
    K_g = -np.eye(ndof)
    boundary_conditions = {'constrained_dofs': []}
    (lam1, mode1) = fcn(K_e, K_g, boundary_conditions, n_nodes)
    c = 5.0
    (lam2, mode2) = fcn(K_e, c * K_g, boundary_conditions, n_nodes)
    assert mode1.shape == (ndof,)
    assert mode2.shape == (ndof,)
    assert abs(lam2 - lam1 / c) < 1e-10
    idx1 = np.argmax(np.abs(mode1))
    idx2 = np.argmax(np.abs(mode2))
    assert idx1 == idx2