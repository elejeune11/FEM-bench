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
    k_exact = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    for n_gauss in [1, 2, 3]:
        k = fcn(x, E, A, n_gauss)
        assert k.shape == (2, 2), f'Incorrect matrix shape for n_gauss={n_gauss}'
        assert np.allclose(k, k.T, rtol=1e-14), f'Matrix not symmetric for n_gauss={n_gauss}'
        assert np.allclose(k, k_exact, rtol=1e-14), f'Incorrect stiffness values for n_gauss={n_gauss}'
        assert abs(np.linalg.det(k)) < 1e-10, f'Matrix should be singular for n_gauss={n_gauss}'
        eigvals = np.linalg.eigvals(k)
        assert np.isclose(min(eigvals), 0, atol=1e-10), 'Should have zero eigenvalue'
        assert max(eigvals) > 0, 'Should have one positive eigenvalue'