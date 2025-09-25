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
    Note: Minor floating-point differences may arise due to roundoff when summing weighted values.
    This test uses a strict but reasonable tolerance to allow for numerical consistency considering the limitations of floating point arithmetic.
    """
    L = 10.0
    E = 200000000000.0
    A = 0.0001
    x_elem = np.array([0.0, L])
    coeff = E * A / L
    expected_K = coeff * np.array([[1, -1], [-1, 1]])
    K_2gauss = fcn(x_elem, E, A, n_gauss=2)
    np.testing.assert_allclose(K_2gauss, expected_K, rtol=1e-09, atol=1e-12, err_msg='Analytical correctness check failed.')
    assert K_2gauss.shape == (2, 2), 'Stiffness matrix must be 2x2.'
    np.testing.assert_allclose(K_2gauss, K_2gauss.T, rtol=1e-09, atol=1e-12, err_msg='Stiffness matrix must be symmetric.')
    det_K = np.linalg.det(K_2gauss)
    assert np.isclose(det_K, 0.0, atol=1e-09), f'Stiffness matrix should be singular, but determinant is {det_K}.'
    K_1gauss = fcn(x_elem, E, A, n_gauss=1)
    K_3gauss = fcn(x_elem, E, A, n_gauss=3)
    np.testing.assert_allclose(K_1gauss, K_2gauss, rtol=1e-09, atol=1e-12, err_msg='1-point and 2-point Gauss integration results differ.')
    np.testing.assert_allclose(K_2gauss, K_3gauss, rtol=1e-09, atol=1e-12, err_msg='2-point and 3-point Gauss integration results differ.')