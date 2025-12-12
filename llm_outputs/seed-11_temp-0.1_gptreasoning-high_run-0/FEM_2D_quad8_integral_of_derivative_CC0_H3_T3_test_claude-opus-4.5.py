def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """Validate ∫_Ω (∇u) dΩ for a cubic field on an identity-mapped Q8 element.
    A 2×2 Gauss–Legendre rule (num_gauss_pts = 4) is exact for this case."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    node_values = np.array([(-1.0) ** 3, 1.0 ** 3, 1.0 ** 3, (-1.0) ** 3, 0.0 ** 3, 1.0 ** 3, 0.0 ** 3, (-1.0) ** 3])
    result = fcn(node_coords, node_values, 4)
    expected_x = 4.0
    expected_y = 0.0
    assert result.shape == (2,)
    assert np.isclose(result[0], expected_x, rtol=1e-10)
    assert np.isclose(result[1], expected_y, atol=1e-10)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """Analytic check with an affine geometric map and a linear field.
    Map the reference square Q = [-1, 1] × [-1, 1] by [x, y]^T = A[ξ, η]^T + c.
    For the linear scalar field u(x, y) = α + βx + γy.
    Test to make sure the function matches the correct analytical solution."""
    A = np.array([[2.0, 0.0], [0.0, 3.0]])
    c = np.array([1.0, 2.0])
    ref_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    node_coords = (A @ ref_coords.T).T + c
    (alpha, beta, gamma) = (5.0, 2.0, 3.0)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    result = fcn(node_coords, node_values, 4)
    area = 4.0 * 6.0
    expected_x = beta * area
    expected_y = gamma * area
    assert result.shape == (2,)
    assert np.isclose(result[0], expected_x, rtol=1e-10)
    assert np.isclose(result[1], expected_y, rtol=1e-10)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """Check quadrature-order sensitivity on a curved, asymmetric Q8 mapping.
    Select a geometry that is intentionally non-affine. With properly selected fixed,
    asymmetric nodal values, the integral ∫_Ω (∇u) dΩ should depend on the
    quadrature order. The test confirms that results from a 3×3 rule differ from those 
    of 1×1 or 2×2, verifying that higher-order integration is sensitive to nonlinear mappings."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.3], [1.2, 0.0], [0.0, 1.1], [-1.1, 0.2]])
    node_values = np.array([1.0, 2.0, 4.0, 3.0, 1.5, 3.0, 3.5, 2.0])
    result_1 = fcn(node_coords, node_values, 1)
    result_4 = fcn(node_coords, node_values, 4)
    result_9 = fcn(node_coords, node_values, 9)
    assert result_1.shape == (2,)
    assert result_4.shape == (2,)
    assert result_9.shape == (2,)
    diff_9_vs_1 = np.linalg.norm(result_9 - result_1)
    diff_9_vs_4 = np.linalg.norm(result_9 - result_4)
    assert diff_9_vs_1 > 1e-06 or diff_9_vs_4 > 1e-06, 'Higher-order quadrature should produce different results for curved geometry'