def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (a, b, c, d, e, f) = (2.0, 1.5, -0.8, 0.7, -1.2, 0.4)
    node_values = a + b * node_coords[:, 0] + c * node_coords[:, 1] + d * node_coords[:, 0] ** 2 + e * node_coords[:, 0] * node_coords[:, 1] + f * node_coords[:, 1] ** 2
    xi_pts = np.array([0.0, 0.5, -0.3])
    eta_pts = np.array([0.0, -0.6, 0.7])
    grad_computed = fcn(node_coords, node_values, xi_pts, eta_pts)
    for (i, (xi, eta)) in enumerate(zip(xi_pts, eta_pts)):
        grad_analytic = np.array([b + 2 * d * xi + e * eta, c + e * xi + 2 * f * eta])
        assert np.allclose(grad_computed[:, i], grad_analytic, rtol=1e-12, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    curved_coords = node_coords + 0.3 * np.array([node_coords[:, 0] ** 2, node_coords[:, 1] ** 2]).T
    (alpha, beta, gamma) = (1.2, -0.7, 0.9)
    node_values = alpha + beta * curved_coords[:, 0] + gamma * curved_coords[:, 1]
    xi_pts = np.array([0.0, 0.5, -0.8])
    eta_pts = np.array([0.0, -0.4, 0.6])
    grad_computed = fcn(curved_coords, node_values, xi_pts, eta_pts)
    expected_grad = np.array([beta, gamma])
    for i in range(len(xi_pts)):
        assert np.allclose(grad_computed[:, i], expected_grad, rtol=1e-12, atol=1e-12)