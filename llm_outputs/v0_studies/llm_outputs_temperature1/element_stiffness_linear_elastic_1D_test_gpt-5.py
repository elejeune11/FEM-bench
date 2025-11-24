def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test checks:
    1. Analytical correctness against (EA/L) * [[1, -1], [-1, 1]]
    2. Shape and symmetry (2x2 and symmetric)
    3. Singularity (zero determinant for an unconstrained single element)
    4. Integration consistency for 1-, 2-, and 3-point Gauss quadrature
    """
    L = 2.7
    E = 123.456
    A = 7.89
    x_elem = np.array([0.0, L], dtype=float)
    scale = E * A / L
    K1 = fcn(x_elem, E, A, 1)
    K2 = fcn(x_elem, E, A, 2)
    K3 = fcn(x_elem, E, A, 3)
    K_expected = scale * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    np.testing.assert_allclose(K2, K_expected, rtol=1e-12, atol=1e-12)
    assert K2.shape == (2, 2)
    np.testing.assert_allclose(K2, K2.T, rtol=0, atol=1e-14)
    det_K2 = np.linalg.det(K2)
    assert abs(det_K2) <= 1e-12 * scale ** 2
    svals = np.linalg.svd(K2, compute_uv=False)
    assert svals[1] <= 1e-12 * svals[0]
    np.testing.assert_allclose(K1, K2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(K3, K2, rtol=1e-12, atol=1e-12)