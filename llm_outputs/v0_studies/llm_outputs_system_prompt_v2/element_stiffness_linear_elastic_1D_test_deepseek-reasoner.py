def test_element_stiffness_comprehensive(fcn):
    """Verify the correctness and robustness of the 1D linear elastic element stiffness matrix."""
    import numpy as np
    x_elem = np.array([0.0, 2.0])
    E = 100.0
    A = 0.5
    L = 2.0
    expected_k = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    k_2pt = fcn(x_elem, E, A, 2)
    assert np.allclose(k_2pt, expected_k, rtol=1e-10, atol=1e-12)
    assert k_2pt.shape == (2, 2)
    assert np.allclose(k_2pt, k_2pt.T, rtol=1e-10, atol=1e-12)
    assert abs(np.linalg.det(k_2pt)) < 1e-10
    k_1pt = fcn(x_elem, E, A, 1)
    k_3pt = fcn(x_elem, E, A, 3)
    assert np.allclose(k_1pt, k_2pt, rtol=1e-10, atol=1e-12)
    assert np.allclose(k_3pt, k_2pt, rtol=1e-10, atol=1e-12)
    x_elem2 = np.array([1.5, 4.0])
    E2 = 200.0
    A2 = 0.25
    L2 = 2.5
    expected_k2 = E2 * A2 / L2 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    k2 = fcn(x_elem2, E2, A2, 2)
    assert np.allclose(k2, expected_k2, rtol=1e-10, atol=1e-12)
    assert np.allclose(k2, k2.T, rtol=1e-10, atol=1e-12)
    assert abs(np.linalg.det(k2)) < 1e-10
    k2_1pt = fcn(x_elem2, E2, A2, 1)
    k2_3pt = fcn(x_elem2, E2, A2, 3)
    assert np.allclose(k2_1pt, k2, rtol=1e-10, atol=1e-12)
    assert np.allclose(k2_3pt, k2, rtol=1e-10, atol=1e-12)