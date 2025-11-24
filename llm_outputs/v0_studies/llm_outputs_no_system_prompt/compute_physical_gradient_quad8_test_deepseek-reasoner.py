def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (a, b, c, d, e, f) = (2, 3, -4, 5, 6, -7)
    xi_nodes = node_coords[:, 0]
    eta_nodes = node_coords[:, 1]
    node_values = a + b * xi_nodes + c * eta_nodes + d * xi_nodes ** 2 + e * xi_nodes * eta_nodes + f * eta_nodes ** 2
    test_points = [(0.0, 0.0), (0.5, -0.3), (-0.7, 0.2)]
    for (xi, eta) in test_points:
        analytic_grad_xi = b + 2 * d * xi + e * eta
        analytic_grad_eta = c + e * xi + 2 * f * eta
        expected_grad = np.array([[analytic_grad_xi], [analytic_grad_eta]])
        computed_grad = fcn(node_coords, node_values, xi, eta)
        assert np.allclose(computed_grad, expected_grad, rtol=1e-10, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[0, 0], [3, 0.5], [2.5, 3], [0.5, 2.5], [1.5, 0.2], [2.8, 1.8], [1.2, 2.8], [0.2, 1.2]])
    (alpha, beta, gamma) = (2, 3, -4)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    expected_grad = np.array([[beta], [gamma]])
    test_points = [(0.0, 0.0), (0.5, -0.5), (-0.3, 0.7), (0.8, 0.2), (-0.9, -0.6)]
    for (xi, eta) in test_points:
        computed_grad = fcn(node_coords, node_values, xi, eta)
        assert np.allclose(computed_grad, expected_grad, rtol=1e-10, atol=1e-12)