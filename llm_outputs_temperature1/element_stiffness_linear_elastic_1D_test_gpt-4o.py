def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    """
    x_elem = np.array([0.0, 1.0])
    E = 210000000000.0
    A = 0.01
    L = x_elem[1] - x_elem[0]
    expected_stiffness = E * A / L * np.array([[1, -1], [-1, 1]])
    for n_gauss in [1, 2, 3]:
        stiffness_matrix = fcn(x_elem, E, A, n_gauss)
        assert np.allclose(stiffness_matrix, expected_stiffness, atol=1e-09), 'Stiffness matrix does not match analytical solution.'
    assert stiffness_matrix.shape == (2, 2), 'Stiffness matrix is not 2x2.'
    assert np.allclose(stiffness_matrix, stiffness_matrix.T, atol=1e-09), 'Stiffness matrix is not symmetric.'
    determinant = np.linalg.det(stiffness_matrix)
    assert np.isclose(determinant, 0, atol=1e-09), 'Stiffness matrix is not singular for an unconstrained element.'
    stiffness_matrix_1 = fcn(x_elem, E, A, 1)
    stiffness_matrix_2 = fcn(x_elem, E, A, 2)
    stiffness_matrix_3 = fcn(x_elem, E, A, 3)
    assert np.allclose(stiffness_matrix_1, stiffness_matrix_2, atol=1e-09), 'Inconsistent results between 1-point and 2-point Gauss quadrature.'
    assert np.allclose(stiffness_matrix_2, stiffness_matrix_3, atol=1e-09), 'Inconsistent results between 2-point and 3-point Gauss quadrature.'