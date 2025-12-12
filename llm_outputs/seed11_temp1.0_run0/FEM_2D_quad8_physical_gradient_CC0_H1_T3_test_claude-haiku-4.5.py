def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    (a, b, c, d, e, f) = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    node_values = np.array([a + b * -1 + c * -1 + d * 1 + e * 1 + f * 1, a + b * 1 + c * -1 + d * 1 + e * -1 + f * 1, a + b * 1 + c * 1 + d * 1 + e * 1 + f * 1, a + b * -1 + c * 1 + d * 1 + e * -1 + f * 1, a + b * 0 + c * -1 + d * 0 + e * 0 + f * 1, a + b * 1 + c * 0 + d * 1 + e * 0 + f * 0, a + b * 0 + c * 1 + d * 0 + e * 0 + f * 1, a + b * -1 + c * 0 + d * 1 + e * 0 + f * 0], dtype=np.float64)
    test_points = [(0.0, 0.0), (0.5, -0.3), (-0.2, 0.7), (0.3, 0.4)]
    for (xi, eta) in test_points:
        grad_phys = fcn(node_coords, node_values, xi, eta)
        expected_du_dxi = b + 2 * d * xi + e * eta
        expected_du_deta = c + e * xi + 2 * f * eta
        assert grad_phys.shape == (2, 1), f'Expected shape (2, 1), got {grad_phys.shape}'
        np.testing.assert_allclose(grad_phys[0, 0], expected_du_dxi, atol=1e-10, err_msg=f'∂u/∂x mismatch at ({xi}, {eta})')
        np.testing.assert_allclose(grad_phys[1, 0], expected_du_deta, atol=1e-10, err_msg=f'∂u/∂y mismatch at ({xi}, {eta})')

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -0.9], [1.1, 1.0], [-0.9, 1.1], [0.0, -1.05], [1.05, 0.0], [0.0, 1.05], [-1.05, 0.0]], dtype=np.float64)
    (alpha, beta, gamma) = (2.0, 3.5, -1.2)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    xi_test = np.array([-0.7, 0.0, 0.5, -0.3])
    eta_test = np.array([-0.5, 0.3, -0.2, 0.8])
    grad_phys = fcn(node_coords, node_values, xi_test, eta_test)
    assert grad_phys.shape == (2, len(xi_test)), f'Expected shape (2, {len(xi_test)}), got {grad_phys.shape}'
    expected_grad = np.array([[beta] * len(xi_test), [gamma] * len(xi_test)])
    np.testing.assert_allclose(grad_phys, expected_grad, atol=1e-09, err_msg='Linear physical field gradient not constant or incorrect')