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
    x_elem = np.array([0.0, 1.0])
    E = 1.0
    A = 1.0
    L = x_elem[1] - x_elem[0]
    K = fcn(x_elem, E, A, 2)
    expected_K = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(K, expected_K, rtol=1e-10, atol=1e-12), f'Stiffness matrix does not match analytical solution. Got:\n{K}\nExpected:\n{expected_K}'
    assert K.shape == (2, 2), f'Expected shape (2, 2), got {K.shape}'
    assert np.allclose(K, K.T, rtol=1e-10, atol=1e-12), 'Stiffness matrix is not symmetric'
    det_K = np.linalg.det(K)
    assert np.isclose(det_K, 0.0, atol=1e-12), f'Stiffness matrix should be singular (det=0), but det={det_K}'
    K_1gauss = fcn(x_elem, E, A, 1)
    K_2gauss = fcn(x_elem, E, A, 2)
    K_3gauss = fcn(x_elem, E, A, 3)
    assert np.allclose(K_1gauss, K_2gauss, rtol=1e-10, atol=1e-12), 'Results differ between 1-point and 2-point Gauss quadrature'
    assert np.allclose(K_2gauss, K_3gauss, rtol=1e-10, atol=1e-12), 'Results differ between 2-point and 3-point Gauss quadrature'
    test_cases = [(np.array([0.0, 2.0]), 2.0, 0.5), (np.array([1.0, 3.0]), 3.0, 2.0), (np.array([-1.0, 1.0]), 1.5, 1.0)]
    for (x_elem_test, E_test, A_test) in test_cases:
        K_test = fcn(x_elem_test, E_test, A_test, 2)
        L_test = x_elem_test[1] - x_elem_test[0]
        expected_K_test = E_test * A_test / L_test * np.array([[1.0, -1.0], [-1.0, 1.0]])
        assert np.allclose(K_test, expected_K_test, rtol=1e-10, atol=1e-12), f'Stiffness matrix incorrect for x_elem={x_elem_test}, E={E_test}, A={A_test}'
        assert np.allclose(K_test, K_test.T, rtol=1e-10, atol=1e-12), f'Stiffness matrix not symmetric for x_elem={x_elem_test}, E={E_test}, A={A_test}'
        assert np.isclose(np.linalg.det(K_test), 0.0, atol=1e-12), f'Stiffness matrix not singular for x_elem={x_elem_test}, E={E_test}, A={A_test}'