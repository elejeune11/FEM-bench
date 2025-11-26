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
    E = 200000000000.0
    A = 0.005
    (x1, x2) = (1.0, 3.5)
    x_elem = np.array([x1, x2])
    L = abs(x2 - x1)
    k_expected = E * A / L
    expected_matrix = k_expected * np.array([[1.0, -1.0], [-1.0, 1.0]])
    K_computed = fcn(x_elem, E, A, n_gauss=2)
    np.testing.assert_allclose(K_computed, expected_matrix, rtol=1e-10, err_msg='Computed stiffness matrix does not match analytical solution (EA/L).')
    assert K_computed.shape == (2, 2), f'Expected shape (2, 2), got {K_computed.shape}'
    np.testing.assert_allclose(K_computed, K_computed.T, rtol=1e-12, err_msg='Stiffness matrix is not symmetric.')
    det = np.linalg.det(K_computed)
    assert abs(det) < 1e-10 * k_expected ** 2, 'Stiffness matrix is not singular (determinant != 0).'
    row_sums = np.sum(K_computed, axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-10 * k_expected, err_msg='Rows do not sum to zero (violation of force equilibrium).')
    K_1 = fcn(x_elem, E, A, n_gauss=1)
    K_2 = fcn(x_elem, E, A, n_gauss=2)
    K_3 = fcn(x_elem, E, A, n_gauss=3)
    np.testing.assert_allclose(K_1, K_2, rtol=1e-12, err_msg='Results differ between 1 and 2 Gauss integration points.')
    np.testing.assert_allclose(K_2, K_3, rtol=1e-12, err_msg='Results differ between 2 and 3 Gauss integration points.')