def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test validates:
    1) Analytical correctness: K = (E*A/L) * [[1, -1], [-1, 1]]
    2) Shape and symmetry: 2x2 and symmetric
    3) Singularity: One zero eigenvalue (reflecting rigid body mode)
    4) Integration consistency: Identical results for 1-, 2-, and 3-point Gauss quadrature
    """
    rtol = 1e-12
    scenarios = [(0.0, 1.0, 210000000000.0, 0.003), (2.5, 5.75, 70000000000.0, 0.01), (-1.2, 0.3, 110000000000.0, 2.5), (0.0, 0.5, 1.0, 1.0)]
    for (x1, x2, E, A) in scenarios:
        L = abs(x2 - x1)
        x = np.array([x1, x2], dtype=float)
        k = E * A / L
        K_expected = k * np.array([[1.0, -1.0], [-1.0, 1.0]])
        for n in (1, 2, 3):
            K = np.asarray(fcn(x, E, A, n), dtype=float)
            assert K.shape == (2, 2)
            assert np.allclose(K, K.T, rtol=rtol, atol=rtol * max(1.0, k))
            assert np.allclose(K, K_expected, rtol=rtol, atol=rtol * max(1.0, k))
        K1 = np.asarray(fcn(x, E, A, 1), dtype=float)
        K2 = np.asarray(fcn(x, E, A, 2), dtype=float)
        K3 = np.asarray(fcn(x, E, A, 3), dtype=float)
        scale = max(1.0, k)
        assert np.allclose(K1, K2, rtol=rtol, atol=rtol * scale)
        assert np.allclose(K2, K3, rtol=rtol, atol=rtol * scale)
        w = np.linalg.eigvalsh(K2)
        lam_max = np.max(np.abs(w))
        lam_min = np.min(np.abs(w))
        assert lam_min <= 1e-12 * max(1.0, lam_max)
        xr = np.array([x2, x1], dtype=float)
        Kr = np.asarray(fcn(xr, E, A, 2), dtype=float)
        assert np.allclose(K2, Kr, rtol=rtol, atol=rtol * scale)