def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. With diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e on the free DOFs; the
    function should return the smallest one and a mode aligned with the
    corresponding DOF, with constrained DOFs set to zero.
    """
    import numpy as np
    import pytest
    n_nodes = 1
    K_e = np.diag([5.0, 2.0, 7.0, 3.0, 9.0, 4.0])
    K_g = -np.eye(6)
    bc = {0: np.array([False, False, True, True, True, True], dtype=bool)}
    (lam, mode) = fcn(K_e, K_g, bc, n_nodes)
    assert lam == pytest.approx(2.0, rel=1e-12, abs=1e-12)
    assert mode.shape == (6,)
    assert np.allclose(mode[2:], 0.0, atol=1e-12)
    idx = np.argmax(np.abs(mode))
    assert idx == 1
    amp = np.abs(mode[idx])
    assert amp > 0
    assert np.abs(mode[0]) <= 1e-12 * amp

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the reduced elastic
    block is singular/ill-conditioned.
    """
    import numpy as np
    import pytest
    n_nodes = 1
    K_e = np.diag([0.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    K_g = -np.eye(6)
    bc = {}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, bc, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """
    import numpy as np
    import pytest
    n_nodes = 1
    K_e = np.zeros((6, 6))
    K_e[:2, :2] = np.array([[0.0, -1.0], [1.0, 0.0]])
    K_g = -np.eye(6)
    bc = {0: np.array([False, False, True, True, True, True], dtype=bool)}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, bc, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    import numpy as np
    import pytest
    n_nodes = 1
    K_e = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    K_g = np.eye(6)
    bc = {}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, bc, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """
    Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size.
    """
    import numpy as np
    import pytest
    n_nodes = 1
    K_e = np.diag([8.0, 5.0, 4.0, 10.0, 9.0, 20.0])
    K_g0 = -np.eye(6)
    bc = {}
    (lam0, mode0) = fcn(K_e, K_g0, bc, n_nodes)
    assert lam0 == pytest.approx(4.0, rel=1e-12, abs=1e-12)
    assert mode0.shape == (6,)
    c = 3.0
    (lam1, mode1) = fcn(K_e, c * K_g0, bc, n_nodes)
    assert lam1 == pytest.approx(lam0 / c, rel=1e-12, abs=1e-12)
    assert mode1.shape == (6,)
    idx_expected = 2
    idx0 = int(np.argmax(np.abs(mode0)))
    idx1 = int(np.argmax(np.abs(mode1)))
    assert idx0 == idx_expected
    assert idx1 == idx_expected