def test_element_stiffness_comprehensive(fcn):
    """Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test checks the following properties of the stiffness matrix computed by
    `element_stiffness_linear_elastic_1D` for a two-node linear element:
    1. Analytical correctness:
          (EA/L) * [[1, -1], [-1, 1]]
    2. Shape and symmetry:
    3. Singularity:
          reflecting rigid body motion.
    4. Integration consistency:
          Gauss quadrature rules when applied to linear elements, since exact integration is achieved.
    Note: Minor floating-point differences may arise due to roundoff when summing weighted values.
    This test uses a strict but reasonable tolerance to allow for numerical consistency considering the limitations of floating point arithmetic.
    """
    x_elem = np.array([0.0, 2.0])
    E = 100.0
    A = 0.5
    L = x_elem[1] - x_elem[0]
    expected_scalar = E * A / L
    expected_matrix = expected_scalar * np.array([[1.0, -1.0], [-1.0, 1.0]])
    K_default = fcn(x_elem, E, A, n_gauss=2)
    np.testing.assert_allclose(K_default, expected_matrix, rtol=1e-10, atol=1e-12)
    assert K_default.shape == (2, 2), 'Stiffness matrix must be 2x2'
    np.testing.assert_allclose(K_default, K_default.T, rtol=1e-10, atol=1e-12)
    det = np.linalg.det(K_default)
    assert abs(det) < 1e-10, f'Stiffness matrix should be singular, but determinant is {det}'
    K_1pt = fcn(x_elem, E, A, n_gauss=1)
    K_2pt = fcn(x_elem, E, A, n_gauss=2)
    K_3pt = fcn(x_elem, E, A, n_gauss=3)
    np.testing.assert_allclose(K_1pt, expected_matrix, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(K_2pt, expected_matrix, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(K_3pt, expected_matrix, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(K_1pt, K_2pt, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(K_1pt, K_3pt, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(K_2pt, K_3pt, rtol=1e-10, atol=1e-12)