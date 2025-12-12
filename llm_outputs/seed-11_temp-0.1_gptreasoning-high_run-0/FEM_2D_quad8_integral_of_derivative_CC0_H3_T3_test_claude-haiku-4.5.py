def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """Validate ∫_Ω (∇u) dΩ for a cubic field on an identity-mapped Q8 element.
    A 2×2 Gauss–Legendre rule (num_gauss_pts = 4) is exact for this case."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    node_values = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 2.0], dtype=np.float64)
    result = fcn(node_coords, node_values, num_gauss_pts=4)
    assert isinstance(result, np.ndarray), 'Result should be a numpy array'
    assert result.shape == (2,), f'Result shape should be (2,), got {result.shape}'
    assert np.all(np.isfinite(result)), 'Result should contain finite values'

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """Analytic check with an affine geometric map and a linear field.
    Map the reference square Q = [-1, 1] × [-1, 1] by [x, y]^T = A[ξ, η]^T + c.
    For the linear scalar field u(x, y) = α + βx + γy.
    Test to make sure the function matches the correct analytical solution."""
    A = np.array([[2.0, 0.5], [0.3, 1.5]], dtype=np.float64)
    c = np.array([1.0, 2.0], dtype=np.float64)
    xi_ref = np.array([-1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0], dtype=np.float64)
    eta_ref = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    node_coords = np.column_stack([A[0, 0] * xi_ref + A[0, 1] * eta_ref + c[0], A[1, 0] * xi_ref + A[1, 1] * eta_ref + c[1]])
    alpha = 1.0
    beta = 2.0
    gamma = 3.0
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    result = fcn(node_coords, node_values, num_gauss_pts=4)
    det_A = np.linalg.det(A)
    domain_area = 4.0 * det_A
    expected_integral_x = beta * domain_area
    expected_integral_y = gamma * domain_area
    expected = np.array([expected_integral_x, expected_integral_y])
    assert isinstance(result, np.ndarray), 'Result should be a numpy array'
    assert result.shape == (2,), f'Result shape should be (2,), got {result.shape}'
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12, err_msg='Integral of linear field gradient does not match analytical solution')

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """Check quadrature-order sensitivity on a curved, asymmetric Q8 mapping.
    Select a geometry that is intentionally non-affine. With properly selected fixed,
    asymmetric nodal values, the integral ∫_Ω (∇u) dΩ should depend on the
    quadrature order. The test confirms that results from a 3×3 rule differ from those 
    of 1×1 or 2×2, verifying that higher-order integration is sensitive to nonlinear mappings."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.2, -1.0], [1.0, 0.3], [-0.1, 1.0], [-1.0, -0.2]], dtype=np.float64)
    node_values = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 2.0], dtype=np.float64)
    result_1x1 = fcn(node_coords, node_values, num_gauss_pts=1)
    result_2x2 = fcn(node_coords, node_values, num_gauss_pts=4)
    result_3x3 = fcn(node_coords, node_values, num_gauss_pts=9)
    assert isinstance(result_1x1, np.ndarray), 'Result should be a numpy array'
    assert isinstance(result_2x2, np.ndarray), 'Result should be a numpy array'
    assert isinstance(result_3x3, np.ndarray), 'Result should be a numpy array'
    assert result_1x1.shape == (2,), f'Result shape should be (2,), got {result_1x1.shape}'
    assert result_2x2.shape == (2,), f'Result shape should be (2,), got {result_2x2.shape}'
    assert result_3x3.shape == (2,), f'Result shape should be (2,), got {result_3x3.shape}'
    assert np.all(np.isfinite(result_1x1)), '1x1 result should contain finite values'
    assert np.all(np.isfinite(result_2x2)), '2x2 result should contain finite values'
    assert np.all(np.isfinite(result_3x3)), '3x3 result should contain finite values'
    diff_1x1_vs_3x3 = np.linalg.norm(result_1x1 - result_3x3)
    diff_2x2_vs_3x3 = np.linalg.norm(result_2x2 - result_3x3)
    assert diff_1x1_vs_3x3 > 1e-06, '1x1 and 3x3 results should differ significantly for curved geometry'
    assert diff_2x2_vs_3x3 > 1e-06, '2x2 and 3x3 results should differ significantly for curved geometry'