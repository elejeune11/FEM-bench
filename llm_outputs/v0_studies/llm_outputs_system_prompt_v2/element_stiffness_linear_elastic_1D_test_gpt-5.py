def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test checks analytical correctness, shape and symmetry, singularity, and integration consistency
    across different Gauss quadrature rules for a two-node linear element.
    """
    E = 210000000000.0
    A = 0.003
    L = 2.5
    x_elem = np.array([1.5, 1.5 + L], dtype=float)
    k1 = fcn(x_elem, E, A, 1)
    k2 = fcn(x_elem, E, A, 2)
    k3 = fcn(x_elem, E, A, 3)
    for K in (k1, k2, k3):
        assert isinstance(K, np.ndarray)
        assert K.shape == (2, 2)
        assert np.allclose(K, K.T, rtol=1e-14, atol=1e-14)
    K_expected = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(k2, K_expected, rtol=1e-12, atol=1e-12)
    det_scale = (E * A / L) ** 2
    det_tol = 1e-12 * det_scale
    for K in (k1, k2, k3):
        detK = np.linalg.det(K)
        assert abs(detK) <= det_tol
    assert np.allclose(k1, k2, rtol=1e-12, atol=1e-12)
    assert np.allclose(k2, k3, rtol=1e-12, atol=1e-12)