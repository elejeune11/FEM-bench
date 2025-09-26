def test_element_stiffness_comprehensive(fcn):
    """Verify the correctness and robustness of the 1D linear elastic element stiffness matrix."""
    x_elem = np.array([0.0, 1.0])
    E = 1.0
    A = 1.0
    L = x_elem[1] - x_elem[0]
    analytical_k = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    k_2gauss = fcn(x_elem, E, A, 2)
    np.testing.assert_allclose(k_2gauss, analytical_k, rtol=1e-14)
    assert k_2gauss.shape == (2, 2)
    np.testing.assert_allclose(k_2gauss, k_2gauss.T, rtol=1e-14)
    assert np.isclose(np.linalg.det(k_2gauss), 0.0)
    k_1gauss = fcn(x_elem, E, A, 1)
    k_3gauss = fcn(x_elem, E, A, 3)
    np.testing.assert_allclose(k_1gauss, analytical_k, rtol=1e-14)
    np.testing.assert_allclose(k_3gauss, analytical_k, rtol=1e-14)
    np.testing.assert_allclose(k_1gauss, k_2gauss, rtol=1e-14)
    np.testing.assert_allclose(k_2gauss, k_3gauss, rtol=1e-14)