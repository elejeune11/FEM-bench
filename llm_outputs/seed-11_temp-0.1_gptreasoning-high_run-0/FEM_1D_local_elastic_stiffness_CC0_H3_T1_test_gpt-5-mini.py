def test_element_stiffness_comprehensive(fcn):
    """Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test checks:
    1. Analytical correctness: K = (E*A/L) * [[1, -1], [-1, 1]] for a two-node linear element.
    2. Shape and symmetry: the returned matrix is 2x2 and symmetric.
    3. Singularity: the element stiffness is singular (rank 1) for an unconstrained single element.
    4. Integration consistency: results with 1-, 2-, and 3-point Gauss quadrature are numerically identical.
    """
    import numpy as np
    x_elem = np.array([1.0, 4.0], dtype=float)
    E = 210.0
    A = 0.02
    L = x_elem[1] - x_elem[0]
    expected = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    K1 = np.asarray(fcn(x_elem, E, A, 1), dtype=float)
    K2 = np.asarray(fcn(x_elem, E, A, 2), dtype=float)
    K3 = np.asarray(fcn(x_elem, E, A, 3), dtype=float)
    rtol = 1e-08
    atol = 1e-12
    assert K1.shape == (2, 2)
    assert np.allclose(K1, K1.T, rtol=rtol, atol=atol)
    assert np.allclose(K1, expected, rtol=rtol, atol=atol)
    assert np.linalg.matrix_rank(K1) == 1
    assert K2.shape == (2, 2)
    assert np.allclose(K2, K2.T, rtol=rtol, atol=atol)
    assert np.allclose(K2, expected, rtol=rtol, atol=atol)
    assert K3.shape == (2, 2)
    assert np.allclose(K3, K3.T, rtol=rtol, atol=atol)
    assert np.allclose(K3, expected, rtol=rtol, atol=atol)
    assert np.allclose(K1, K2, rtol=rtol, atol=atol)
    assert np.allclose(K2, K3, rtol=rtol, atol=atol)