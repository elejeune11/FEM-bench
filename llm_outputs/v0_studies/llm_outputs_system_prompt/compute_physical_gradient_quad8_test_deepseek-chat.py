def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (a, b, c, d, e, f) = (2.0, 1.5, -0.8, 0.7, -1.2, 0.9)
    node_values = np.array([a + b * -1 + c * -1 + d * (-1) ** 2 + e * -1 * -1 + f * (-1) ** 2, a + b * 1 + c * -1 + d * 1 ** 2 + e * 1 * -1 + f * (-1) ** 2, a + b * 1 + c * 1 + d * 1 ** 2 + e * 1 * 1 + f * 1 ** 2, a + b * -1 + c * 1 + d * (-1) ** 2 + e * -1 * 1 + f * 1 ** 2, a + b * 0 + c * -1 + d * 0 ** 2 + e * 0 * -1 + f * (-1) ** 2, a + b * 1 + c * 0 + d * 1 ** 2 + e * 1 * 0 + f * 0 ** 2, a + b * 0 + c * 1 + d * 0 ** 2 + e * 0 * 1 + f * 1 ** 2, a + b * -1 + c * 0 + d * (-1) ** 2 + e * -1 * 0 + f * 0 ** 2])
    xi_pts = np.array([0.0, 0.5, -0.3])
    eta_pts = np.array([0.0, -0.7, 0.4])
    grad_computed = fcn(node_coords, node_values, xi_pts, eta_pts)
    for (i, (xi, eta)) in enumerate(zip(xi_pts, eta_pts)):
        analytic_grad = np.array([b + 2 * d * xi + e * eta, c + e * xi + 2 * f * eta])
        assert np.allclose(grad_computed[:, i], analytic_grad, rtol=1e-10)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    curved_coords = node_coords + 0.3 * np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]])
    (alpha, beta, gamma) = (2.0, 1.5, -0.8)
    node_values = alpha + beta * curved_coords[:, 0] + gamma * curved_coords[:, 1]
    xi_pts = np.array([0.0, 0.5, -0.7, 0.3, -0.9])
    eta_pts = np.array([0.0, -0.5, 0.6, -0.2, 0.8])
    grad_computed = fcn(curved_coords, node_values, xi_pts, eta_pts)
    expected_grad = np.array([beta, gamma])
    for i in range(len(xi_pts)):
        assert np.allclose(grad_computed[:, i], expected_grad, rtol=1e-10)