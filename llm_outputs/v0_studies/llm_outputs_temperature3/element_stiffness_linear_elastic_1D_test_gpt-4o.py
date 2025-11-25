def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test checks analytical correctness, shape and symmetry, singularity, and integration consistency.
    """
    x_elem = np.array([0.0, 1.0])
    E = 210000000000.0
    A = 0.01
    expected_stiffness_matrix = E * A / 1.0 * np.array([[1, -1], [-1, 1]])
    for n_gauss in [1, 2, 3]:
        stiffness_matrix = fcn(x_elem, E, A, n_gauss)
        assert np.allclose(stiffness_matrix, expected_stiffness_matrix, atol=1e-09), f'Stiffness matrix does not match expected result for n_gauss={n_gauss}'
        assert stiffness_matrix.shape == (2, 2), 'Stiffness matrix is not 2x2'
        assert np.allclose(stiffness_matrix, stiffness_matrix.T, atol=1e-09), 'Stiffness matrix is not symmetric'
        assert np.isclose(np.linalg.det(stiffness_matrix), 0.0, atol=1e-09), 'Stiffness matrix is not singular'