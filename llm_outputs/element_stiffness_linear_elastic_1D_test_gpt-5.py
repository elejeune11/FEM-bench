def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test checks:
    1) Analytical correctness: K = (E*A/L) * [[1, -1], [-1, 1]]
    2) Shape and symmetry: 2x2 and symmetric
    3) Singularity: one zero eigenvalue (rigid body mode)
    4) Integration consistency: identical results for 1-, 2-, and 3-point Gauss quadrature
    """
    L = 3.7
    E = 123456000000.0
    A = 0.0123
    x0 = 1.1
    x_elem = np.array([x0, x0 + L])
    k = E * A / L
    K_expected = k * np.array([[1.0, -1.0], [-1.0, 1.0]])
    K1 = fcn(x_elem, E, A, 1)
    K2 = fcn(x_elem, E, A, 2)
    K3 = fcn(x_elem, E, A, 3)
    rtol = 1e-12
    atol = 1e-12 * max(1.0, abs(k))
    assert K2.shape == (2, 2)
    assert np.allclose(K2, K2.T, rtol=rtol, atol=atol)
    assert np.allclose(K1, K_expected, rtol=rtol, atol=atol)
    assert np.allclose(K2, K_expected, rtol=rtol, atol=atol)
    assert np.allclose(K3, K_expected, rtol=rtol, atol=atol)
    eigvals = np.linalg.eigvalsh(K2)
    eigvals.sort()
    assert abs(eigvals[0]) <= 1e-09 * max(1.0, abs(k))
    assert eigvals[1] > 1e-12 * max(1.0, abs(k))
    assert np.allclose(K1, K2, rtol=rtol, atol=atol)
    assert np.allclose(K2, K3, rtol=rtol, atol=atol)
    assert np.allclose(K1, K3, rtol=rtol, atol=atol)