def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test validates:
    1) Analytical correctness against (EA/L) * [[1, -1], [-1, 1]]
    2) Shape (2x2) and symmetry
    3) Singularity (zero determinant) for an unconstrained single element
    4) Integration consistency across 1-, 2-, and 3-point Gauss rules
    """
    tol = 1e-12
    test_cases = [(np.array([0.0, 2.5]), 70.0, 3.0), (np.array([1.2, 4.8]), 123.4, 0.56), (np.array([5.0, 1.0]), 80.0, 2.5)]
    for (x_elem, E, A) in test_cases:
        L = abs(x_elem[1] - x_elem[0])
        assert L > 0.0
        K1 = fcn(x_elem, E, A, 1)
        K2 = fcn(x_elem, E, A, 2)
        K3 = fcn(x_elem, E, A, 3)
        assert isinstance(K2, np.ndarray)
        assert K2.shape == (2, 2)
        assert np.isfinite(K2).all()
        assert np.allclose(K2, K2.T, rtol=tol, atol=tol)
        K_expected = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
        assert np.allclose(K2, K_expected, rtol=1e-12, atol=1e-12)
        det = np.linalg.det(K2)
        scale = (E * A / L) ** 2
        assert abs(det) <= 1e-10 * scale
        assert np.allclose(K2.sum(axis=1), np.zeros(2), rtol=1e-12, atol=1e-12)
        assert np.allclose(K2.sum(axis=0), np.zeros(2), rtol=1e-12, atol=1e-12)
        assert np.allclose(K1, K2, rtol=1e-12, atol=1e-12)
        assert np.allclose(K3, K2, rtol=1e-12, atol=1e-12)