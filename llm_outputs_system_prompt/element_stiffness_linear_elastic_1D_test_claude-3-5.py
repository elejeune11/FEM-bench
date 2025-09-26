def test_element_stiffness_comprehensive(fcn):
    """Verify the correctness and robustness of the 1D linear elastic element stiffness matrix."""
    import numpy as np
    E = 200000000000.0
    A = 0.01
    L = 0.5
    x_elem = np.array([0.0, L])
    tol = 1e-12
    K_exact = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    for n_gauss in [1, 2, 3]:
        K = fcn(x_elem, E, A, n_gauss)
        assert K.shape == (2, 2)
        assert np.allclose(K, K.T, rtol=tol)
        assert np.allclose(K, K_exact, rtol=tol)
        assert abs(np.linalg.det(K)) < tol
        eigenvals = np.linalg.eigvals(K)
        assert np.all(eigenvals >= -tol)
        assert np.any(abs(eigenvals) < tol)
    (E2, A2) = (2 * E, 3 * A)
    K2 = fcn(x_elem, E2, A2, 2)
    assert np.allclose(K2, E2 * A2 / (E * A) * K_exact, rtol=tol)
    L2 = 2 * L
    x_elem2 = np.array([0.0, L2])
    K3 = fcn(x_elem2, E, A, 2)
    assert np.allclose(K3, L / L2 * K_exact, rtol=tol)