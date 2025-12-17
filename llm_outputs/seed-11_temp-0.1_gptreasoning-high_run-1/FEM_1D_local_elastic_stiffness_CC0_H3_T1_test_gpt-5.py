def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test checks the following properties:
    1. Analytical correctness: K = (E*A/L) * [[1, -1], [-1, 1]]
    2. Shape and symmetry: 2x2 and symmetric
    3. Singularity: determinant is (numerically) zero for an unconstrained single element
    4. Integration consistency: results are identical for 1-, 2-, and 3-point Gauss quadrature
    """
    import numpy as np
    E = 123.456
    A = 0.789
    x_elem = np.array([1.25, 3.75], dtype=float)
    L = float(abs(x_elem[1] - x_elem[0]))
    k = E * A / L
    K_expected = k * np.array([[1.0, -1.0], [-1.0, 1.0]])
    K1 = fcn(x_elem, E, A, 1)
    K2 = fcn(x_elem, E, A, 2)
    K3 = fcn(x_elem, E, A, 3)
    assert isinstance(K2, np.ndarray)
    assert K2.shape == (2, 2)
    assert np.allclose(K1, K1.T, rtol=1e-12, atol=1e-14)
    assert np.allclose(K2, K2.T, rtol=1e-12, atol=1e-14)
    assert np.allclose(K3, K3.T, rtol=1e-12, atol=1e-14)
    assert np.allclose(K1, K_expected, rtol=1e-12, atol=1e-12)
    assert np.allclose(K2, K_expected, rtol=1e-12, atol=1e-12)
    assert np.allclose(K3, K_expected, rtol=1e-12, atol=1e-12)
    det = np.linalg.det(K2)
    tol_det = np.finfo(float).eps * k ** 2 * 10.0 + 1e-15
    assert abs(det) <= tol_det
    evals = np.linalg.eigvalsh(K2)
    max_eval = float(np.max(np.abs(evals)))
    assert np.isclose(evals.min(), 0.0, atol=1e-12 * max_eval + 1e-15)
    assert np.allclose(K1, K2, rtol=1e-12, atol=1e-12)
    assert np.allclose(K2, K3, rtol=1e-12, atol=1e-12)