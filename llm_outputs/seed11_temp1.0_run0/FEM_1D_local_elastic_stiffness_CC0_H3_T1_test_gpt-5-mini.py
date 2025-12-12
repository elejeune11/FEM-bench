def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test checks:
    1. Analytical correctness: for an element of length L with modulus E and area A the
       stiffness matrix equals (EA/L) * [[1, -1], [-1, 1]].
    2. Shape and symmetry: returned matrix is 2x2 and symmetric.
    3. Singularity: determinant is effectively zero (rigid body mode present).
    4. Integration consistency: results are numerically identical for 1-, 2-, and 3-point
       Gauss quadrature within a reasonable tolerance for linear elements.
    """
    L = 2.5
    x_elem = np.array([0.0, L], dtype=float)
    E = 210000000000.0
    A = 0.003
    factor = E * A / L
    expected = factor * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    K2 = fcn(x_elem, E, A, 2)
    assert isinstance(K2, np.ndarray)
    assert K2.shape == (2, 2)
    assert np.allclose(K2, K2.T, rtol=0, atol=1e-12)
    assert np.allclose(K2, expected, rtol=1e-09, atol=1e-12)
    det_K2 = float(np.linalg.det(K2))
    scale = factor if factor != 0 else 1.0
    assert abs(det_K2) < 1e-08 * scale * scale
    K1 = fcn(x_elem, E, A, 1)
    K3 = fcn(x_elem, E, A, 3)
    assert np.allclose(K1, K2, rtol=1e-09, atol=1e-12)
    assert np.allclose(K3, K2, rtol=1e-09, atol=1e-12)