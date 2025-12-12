def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (a, b, c, d, e, f) = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    node_values = np.array([a + b * -1 + c * -1 + d * 1 + e * 1 + f * 1, a + b * 1 + c * -1 + d * 1 + e * -1 + f * 1, a + b * 1 + c * 1 + d * 1 + e * 1 + f * 1, a + b * -1 + c * 1 + d * 1 + e * -1 + f * 1, a + b * 0 + c * -1 + d * 0 + e * 0 + f * 1, a + b * 1 + c * 0 + d * 1 + e * 0 + f * 0, a + b * 0 + c * 1 + d * 0 + e * 0 + f * 1, a + b * -1 + c * 0 + d * 1 + e * 0 + f * 0])
    test_points = [(0.0, 0.0), (0.5, 0.3), (-0.7, 0.2), (0.1, -0.8)]
    for (xi, eta) in test_points:
        grad_phys = fcn(node_coords, node_values, xi, eta)
        du_dxi = b + 2 * d * xi + e * eta
        du_deta = c + e * xi + 2 * f * eta
        expected_grad = np.array([[du_dxi], [du_deta]])
        np.testing.assert_allclose(grad_phys, expected_grad, atol=1e-10, err_msg=f'Gradient mismatch at (ξ={xi}, η={eta})')

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -0.8], [1.2, 1.0], [-0.8, 1.2], [0.0, -1.1], [1.1, -0.1], [0.1, 1.1], [-1.1, 0.0]])
    (alpha, beta, gamma) = (5.0, 2.0, 3.0)
    node_values = np.array([alpha + beta * node_coords[i, 0] + gamma * node_coords[i, 1] for i in range(8)])
    test_points = [(0.0, 0.0), (0.5, 0.5), (-0.5, 0.3), (0.2, -0.7), (-0.8, -0.8)]
    expected_grad = np.array([[beta], [gamma]])
    for (xi, eta) in test_points:
        grad_phys = fcn(node_coords, node_values, xi, eta)
        np.testing.assert_allclose(grad_phys, expected_grad, atol=1e-09, err_msg=f'Linear field gradient mismatch at (ξ={xi}, η={eta})')