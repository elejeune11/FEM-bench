def test_element_stiffness_comprehensive(fcn):
    """Verify the correctness and robustness of the 1D linear elastic element stiffness matrix."""
    E = 200000000000.0
    A = 0.01
    L = 0.5
    x_elem = np.array([0.0, L])
    k_expected = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    for n_gauss in [1, 2, 3]:
        k_computed = fcn(x_elem, E, A, n_gauss)
        assert k_computed.shape == (2, 2), 'Stiffness matrix must be 2x2'
        assert np.allclose(k_computed, k_computed.T, rtol=1e-14), 'Stiffness matrix must be symmetric'
        assert np.allclose(k_computed, k_expected, rtol=1e-14), f'Incorrect stiffness matrix for {n_gauss} Gauss points'
        assert abs(np.linalg.det(k_computed)) < 1e-10, 'Stiffness matrix should be singular'
        eigvals = np.linalg.eigvals(k_computed)
        assert np.sum(np.abs(eigvals) < 1e-10) == 1, 'Should have exactly one zero eigenvalue'
    lengths = [0.1, 1.0, 10.0]
    for L in lengths:
        x_elem = np.array([0.0, L])
        k_expected = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
        k_computed = fcn(x_elem, E, A, 2)
        assert np.allclose(k_computed, k_expected, rtol=1e-14), f'Incorrect stiffness matrix for length {L}'
    moduli = [70000000000.0, 200000000000.0]
    areas = [0.001, 0.01]
    for E in moduli:
        for A in areas:
            k_expected = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
            k_computed = fcn(x_elem, E, A, 2)
            assert np.allclose(k_computed, k_expected, rtol=1e-14), f'Incorrect stiffness matrix for E={E}, A={A}'