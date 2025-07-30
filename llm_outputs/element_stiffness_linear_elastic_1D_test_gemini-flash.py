def test_element_stiffness_comprehensive(fcn):
    """Verify the correctness and robustness of the 1D linear elastic element stiffness matrix."""
    x_elem = np.array([0, 1])
    E = 1.0
    A = 1.0
    L = x_elem[1] - x_elem[0]
    K_analytical = E * A / L * np.array([[1, -1], [-1, 1]])
    K_2gauss = fcn(x_elem, E, A, 2)
    np.testing.assert_allclose(K_2gauss, K_analytical, rtol=1e-14)
    assert K_2gauss.shape == (2, 2)
    np.testing.assert_allclose(K_2gauss, K_2gauss.T, rtol=1e-14)
    assert np.isclose(np.linalg.det(K_2gauss), 0)
    K_1gauss = fcn(x_elem, E, A, 1)
    K_3gauss = fcn(x_elem, E, A, 3)
    np.testing.assert_allclose(K_1gauss, K_analytical, rtol=1e-14)
    np.testing.assert_allclose(K_3gauss, K_analytical, rtol=1e-14)