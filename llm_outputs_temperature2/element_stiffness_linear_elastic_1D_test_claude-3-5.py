def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    Tests analytical correctness, shape/symmetry, singularity, and integration consistency.
    """
    import numpy as np
    E = 200000000000.0
    A = 0.01
    L = 1.0
    x = np.array([0.0, L])
    k_exact = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    k = fcn(x, E, A, n_gauss=2)
    assert np.allclose(k, k_exact, rtol=1e-14, atol=1e-14)
    assert k.shape == (2, 2)
    assert np.allclose(k, k.T, rtol=1e-14, atol=1e-14)
    assert abs(np.linalg.det(k)) < 1e-14
    k1 = fcn(x, E, A, n_gauss=1)
    k2 = fcn(x, E, A, n_gauss=2)
    k3 = fcn(x, E, A, n_gauss=3)
    assert np.allclose(k1, k2, rtol=1e-14, atol=1e-14)
    assert np.allclose(k2, k3, rtol=1e-14, atol=1e-14)