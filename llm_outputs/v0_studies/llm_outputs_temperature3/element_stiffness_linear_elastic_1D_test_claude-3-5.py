def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    Tests analytical correctness, shape/symmetry, singularity, and integration consistency
    for different Gauss quadrature orders.
    """
    import numpy as np
    E = 200000000000.0
    A = 0.01
    L = 1.0
    x = np.array([0.0, L])
    tol = 1e-12
    K_exact = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    K = fcn(x, E, A, n_gauss=2)
    assert np.allclose(K, K_exact, rtol=tol, atol=tol)
    assert K.shape == (2, 2)
    assert np.allclose(K, K.T, rtol=tol, atol=tol)
    assert abs(np.linalg.det(K)) < tol
    K1 = fcn(x, E, A, n_gauss=1)
    K2 = fcn(x, E, A, n_gauss=2)
    K3 = fcn(x, E, A, n_gauss=3)
    assert np.allclose(K1, K2, rtol=tol, atol=tol)
    assert np.allclose(K2, K3, rtol=tol, atol=tol)