def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """Validate ∫_Ω (∇u) dΩ for a cubic field on an identity-mapped Q8 element.
    A 2×2 Gauss–Legendre rule (num_gauss_pts = 4) is exact for this case."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    node_values = np.array([0.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float64)
    result = fcn(node_coords, node_values, num_gauss_pts=4)
    assert isinstance(result, np.ndarray), 'Result should be numpy array'
    assert result.shape == (2,), 'Result shape should be (2,)'
    assert np.all(np.isfinite(result)), 'Result should contain finite values'

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """Analytic check with an affine geometric map and a linear field.
    Map the reference square Q = [-1, 1] × [-1, 1] by [x, y]^T = A[ξ, η]^T + c.
    For the linear scalar field u(x, y) = α + βx + γy.
    Test to make sure the function matches the correct analytical solution."""
    A = np.array([[2.0, 0.5], [0.0, 1.5]], dtype=np.float64)
    c = np.array([1.0, 2.0], dtype=np.float64)
    (alpha, beta, gamma) = (1.0, 2.0, 3.0)
    node_coords_ref = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    node_coords = node_coords_ref @ A.T + c
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    result = fcn(node_coords, node_values, num_gauss_pts=4)
    det_A = np.linalg.det(A)
    domain_area = 4.0 * det_A
    expected = np.array([beta * domain_area, gamma * domain_area], dtype=np.float64)
    assert result.shape == (2,), 'Result shape should be (2,)'
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12, err_msg='Integral of linear field gradient should match analytical solution')

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """Check quadrature-order sensitivity on a curved, asymmetric Q8 mapping.
    Select a geometry that is intentionally non-affine. With properly selected fixed,
    asymmetric nodal values, the integral ∫_Ω (∇u) dΩ should depend on the
    quadrature order. The test confirms that results from a 3×3 rule differ from those 
    of 1×1 or 2×2, verifying that higher-order integration is sensitive to nonlinear mappings."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.2, 1.2], [-1.1, 1.0], [0.0, -0.9], [1.05, 0.1], [0.1, 1.1], [-1.05, 0.0]], dtype=np.float64)
    node_values = np.array([1.0, 2.0, 3.5, 1.5, 1.5, 2.5, 2.0, 1.2], dtype=np.float64)
    result_1x1 = fcn(node_coords, node_values, num_gauss_pts=1)
    result_2x2 = fcn(node_coords, node_values, num_gauss_pts=4)
    result_3x3 = fcn(node_coords, node_values, num_gauss_pts=9)
    assert result_1x1.shape == (2,), '1x1 result shape should be (2,)'
    assert result_2x2.shape == (2,), '2x2 result shape should be (2,)'
    assert result_3x3.shape == (2,), '3x3 result shape should be (2,)'
    diff_1x1_vs_3x3 = np.linalg.norm(result_1x1 - result_3x3)
    diff_2x2_vs_3x3 = np.linalg.norm(result_2x2 - result_3x3)
    assert diff_1x1_vs_3x3 > 1e-08, '1x1 and 3x3 results should differ on curved non-affine mapping'
    assert diff_2x2_vs_3x3 > 1e-08, '2x2 and 3x3 results should differ on curved non-affine mapping'