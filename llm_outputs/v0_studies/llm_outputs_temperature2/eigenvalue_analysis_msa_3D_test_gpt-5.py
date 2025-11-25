def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. With diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e. The function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 1
    Ke = np.diag([2.0, 5.0, 1.0, 3.0, 4.0, 6.0])
    Kg = -np.eye(6)
    bc = None
    (lam, vec) = fcn(Ke, Kg, bc, n_nodes)
    assert np.isclose(lam, 1.0, rtol=1e-08, atol=1e-12)
    assert vec.shape == (6,)
    idx = np.argmax(np.abs(vec))
    assert idx == 2
    other = np.delete(vec, idx)
    assert np.all(np.abs(other) <= 1e-08 * np.max(np.abs(vec)))

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic or geometric block is singular/ill-conditioned.
    Here we make K_g singular (zero matrix) to trigger the failure.
    """
    n_nodes = 1
    Ke = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    Kg = np.zeros((6, 6))
    bc = None
    with pytest.raises(ValueError):
        fcn(Ke, Kg, bc, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs. Construct K_g with
    2x2 rotation blocks leading to purely imaginary eigenvalues.
    """
    n_nodes = 1
    Ke = np.eye(6)
    R = np.array([[0.0, 1.0], [-1.0, 0.0]])
    Kg = np.block([[R, np.zeros((2, 2)), np.zeros((2, 2))], [np.zeros((2, 2)), R, np.zeros((2, 2))], [np.zeros((2, 2)), np.zeros((2, 2)), R]])
    bc = None
    with pytest.raises(ValueError):
        fcn(Ke, Kg, bc, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present. With K_g = +I and positive definite K_e,
    all generalized eigenvalues are negative.
    """
    n_nodes = 1
    Ke = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    Kg = np.eye(6)
    bc = None
    with pytest.raises(ValueError):
        fcn(Ke, Kg, bc, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """
    Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size.
    """
    n_nodes = 1
    Ke = np.diag([5.0, 2.0, 7.0, 3.0, 6.0, 4.0])
    Kg = -np.eye(6)
    bc = None
    (lam1, v1) = fcn(Ke, Kg, bc, n_nodes)
    c = 3.7
    (lam2, v2) = fcn(Ke, c * Kg, bc, n_nodes)
    assert np.isclose(lam2, lam1 / c, rtol=1e-09, atol=1e-12)
    assert v1.shape == (6,)
    assert v2.shape == (6,)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    assert n1 > 0 and n2 > 0
    cosang = float(np.abs(np.dot(v1 / n1, v2 / n2)))
    assert cosang > 1 - 1e-07