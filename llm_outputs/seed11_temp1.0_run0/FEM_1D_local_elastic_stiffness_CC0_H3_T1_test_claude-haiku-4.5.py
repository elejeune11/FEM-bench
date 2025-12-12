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
    x_elem = np.array([0.0, 1.0])
    E = 1.0
    A = 1.0
    K = fcn(x_elem, E, A, n_gauss=2)
    expected = np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert K.shape == (2, 2), 'Stiffness matrix must be 2x2'
    np.testing.assert_allclose(K, expected, rtol=1e-10, atol=1e-12)
    assert np.allclose(K, K.T, rtol=1e-10, atol=1e-12), 'Stiffness matrix must be symmetric'
    det_K = np.linalg.det(K)
    assert np.abs(det_K) < 1e-10, 'Stiffness matrix must be singular (det ≈ 0)'
    x_elem_2 = np.array([0.0, 2.0])
    K_2 = fcn(x_elem_2, E, A, n_gauss=2)
    expected_2 = np.array([[0.5, -0.5], [-0.5, 0.5]])
    np.testing.assert_allclose(K_2, expected_2, rtol=1e-10, atol=1e-12)
    E_test = 2.5
    A_test = 3.0
    K_3 = fcn(x_elem, E_test, A_test, n_gauss=2)
    expected_3 = np.array([[7.5, -7.5], [-7.5, 7.5]])
    np.testing.assert_allclose(K_3, expected_3, rtol=1e-10, atol=1e-12)
    K_gauss1 = fcn(x_elem, E, A, n_gauss=1)
    K_gauss2 = fcn(x_elem, E, A, n_gauss=2)
    K_gauss3 = fcn(x_elem, E, A, n_gauss=3)
    np.testing.assert_allclose(K_gauss1, K_gauss2, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(K_gauss2, K_gauss3, rtol=1e-10, atol=1e-12)
    x_elem_offset = np.array([5.0, 8.0])
    K_offset = fcn(x_elem_offset, E, A, n_gauss=2)
    expected_offset = np.array([[1.0 / 3.0, -1.0 / 3.0], [-1.0 / 3.0, 1.0 / 3.0]])
    np.testing.assert_allclose(K_offset, expected_offset, rtol=1e-10, atol=1e-12)
    E_large = 1000000.0
    A_large = 0.0001
    K_large = fcn(x_elem, E_large, A_large, n_gauss=2)
    expected_large = np.array([[100.0, -100.0], [-100.0, 100.0]])
    np.testing.assert_allclose(K_large, expected_large, rtol=1e-08, atol=1e-10)