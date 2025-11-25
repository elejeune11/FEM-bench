def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (a, b, c, d, e, f) = (2.0, 1.5, -0.8, 0.7, -0.3, 0.4)

    def u_func(xi, eta):
        return a + b * xi + c * eta + d * xi ** 2 + e * xi * eta + f * eta ** 2

    def du_dxi(xi, eta):
        return b + 2 * d * xi + e * eta

    def du_deta(xi, eta):
        return c + e * xi + 2 * f * eta
    node_values = np.array([u_func(xi, eta) for (xi, eta) in node_coords])
    test_points = [(0, 0), (0.5, -0.3), (-0.7, 0.2), (0.8, 0.9)]
    for (xi, eta) in test_points:
        grad_computed = fcn(node_coords, node_values, xi, eta)
        grad_analytic = np.array([du_dxi(xi, eta), du_deta(xi, eta)])
        np.testing.assert_allclose(grad_computed.flatten(), grad_analytic, rtol=1e-10, atol=1e-10, err_msg=f'Gradient mismatch at (ξ={xi}, η={eta})')

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[0, 0], [3, 0.5], [2.5, 3], [0.5, 2.5], [1.5, 0.2], [2.8, 1.5], [1.2, 2.8], [0.2, 1.2]])
    (alpha, beta, gamma) = (2.0, 1.5, -0.8)
    node_values = np.array([alpha + beta * x + gamma * y for (x, y) in node_coords])
    expected_grad = np.array([[beta], [gamma]])
    test_points = [(0, 0), (-0.5, -0.5), (0.7, -0.3), (-0.2, 0.6), (0.4, 0.4)]
    for (xi, eta) in test_points:
        grad_computed = fcn(node_coords, node_values, xi, eta)
        np.testing.assert_allclose(grad_computed, expected_grad, rtol=1e-10, atol=1e-10, err_msg=f'Constant gradient mismatch at (ξ={xi}, η={eta})')