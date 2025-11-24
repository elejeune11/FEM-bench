def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (a, b, c, d, e, f) = (2.0, 1.5, -0.8, 0.7, -1.2, 0.9)
    node_values = np.array([a + b * x + c * y + d * x ** 2 + e * x * y + f * y ** 2 for (x, y) in node_coords])
    test_points = [(0.0, 0.0), (0.5, -0.3), (-0.7, 0.4)]
    for (xi, eta) in test_points:
        analytic_grad = np.array([b + 2 * d * xi + e * eta, c + e * xi + 2 * f * eta])
        computed_grad = fcn(node_coords, node_values, xi, eta)
        np.testing.assert_allclose(computed_grad.flatten(), analytic_grad, rtol=1e-10)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[0, 0], [3, 0.5], [2.5, 3], [-0.5, 2], [1.5, 0.2], [2.8, 1.8], [1.0, 2.5], [-0.3, 1.0]])
    (alpha, beta, gamma) = (2.0, 1.5, -0.8)
    node_values = np.array([alpha + beta * x + gamma * y for (x, y) in node_coords])
    expected_grad = np.array([beta, gamma])
    test_points = [(0.0, 0.0), (0.5, -0.5), (-0.7, 0.3), (0.2, 0.6)]
    for (xi, eta) in test_points:
        computed_grad = fcn(node_coords, node_values, xi, eta)
        np.testing.assert_allclose(computed_grad.flatten(), expected_grad, rtol=1e-10)