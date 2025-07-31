def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test checks the following properties of the stiffness matrix computed by
    `element_stiffness_linear_elastic_1D` for a two-node linear element:
    1. Analytical correctness:
          (EA/L) * [[1, -1], [-1, 1]]
    2. Shape and symmetry:
    3. Singularity:
          reflecting rigid body motion.
    4. Integration consistency:
          Gauss quadrature rules when applied to linear elements, since exact integration is achieved.
    """
    x_elem = np.array([0.0, 1.0])
    E = 2.0
    A = 3.0
    L = x_elem[1] - x_elem[0]
    expected_matrix = E * A / L * np.array([[1, -1], [-1, 1]])
    computed_matrix = fcn(x_elem, E, A, n_gauss=2)
    np.testing.assert_allclose(computed_matrix, expected_matrix, rtol=1e-10)
    assert computed_matrix.shape == (2, 2)
    np.testing.assert_allclose(computed_matrix, computed_matrix.T, rtol=1e-10)
    assert np.linalg.det(computed_matrix) < 1e-10
    matrix_1pt = fcn(x_elem, E, A, n_gauss=1)
    matrix_2pt = fcn(x_elem, E, A, n_gauss=2)
    matrix_3pt = fcn(x_elem, E, A, n_gauss=3)
    np.testing.assert_allclose(matrix_1pt, matrix_2pt, rtol=1e-10)
    np.testing.assert_allclose(matrix_1pt, matrix_3pt, rtol=1e-10)