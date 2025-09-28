def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (a, b, c, d, e, f) = (2.0, 1.5, -0.8, 0.7, -0.3, 0.4)

    def quadratic_func(xi, eta):
        return a + b * xi + c * eta + d * xi ** 2 + e * xi * eta + f * eta ** 2

    def analytic_gradient(xi, eta):
        du_dxi = b + 2 * d * xi + e * eta
        du_deta = c + e * xi + 2 * f * eta
        return np.array([du_dxi, du_deta])
    node_values = np.array([quadratic_func(xi, eta) for (xi, eta) in node_coords])
    test_points = [(0.0, 0.0), (0.5, -0.3), (-0.7, 0.2), (0.9, 0.8)]
    for (xi, eta) in test_points:
        computed_grad = fcn(node_coords, node_values, xi, eta)
        expected_grad = analytic_gradient(xi, eta)
        np.testing.assert_allclose(computed_grad.flatten(), expected_grad, rtol=1e-10, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[0, 0], [3, -1], [4, 3], [1, 2], [1.5, -0.5], [3.5, 1], [2.5, 2.5], [0.5, 1]])
    (alpha, beta, gamma) = (2.0, 1.5, -0.8)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    expected_grad = np.array([beta, gamma])
    test_points = [(0.0, 0.0), (0.5, -0.5), (-0.7, 0.3), (0.2, 0.6), (0.9, -0.8), (-0.3, -0.9)]
    for (xi, eta) in test_points:
        computed_grad = fcn(node_coords, node_values, xi, eta)
        np.testing.assert_allclose(computed_grad.flatten(), expected_grad, rtol=1e-10, atol=1e-12)