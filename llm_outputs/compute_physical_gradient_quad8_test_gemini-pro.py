def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient."""
    (a, b, c, d, e, f) = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    xi = np.array([-0.5, 0.0, 0.5])
    eta = np.array([0.25, 0.5, 0.75])
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    node_values = a + b * node_coords[:, 0] + c * node_coords[:, 1] + d * node_coords[:, 0] ** 2 + e * node_coords[:, 0] * node_coords[:, 1] + f * node_coords[:, 1] ** 2
    grad_phys = fcn(node_coords, node_values, xi, eta)
    grad_analytic = np.array([b + 2 * d * xi + e * eta, c + e * xi + 2 * f * eta])
    np.testing.assert_allclose(grad_phys, grad_analytic, rtol=1e-12, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    (alpha, beta, gamma) = (1.0, 2.0, 3.0)
    xi = np.array([-0.5, 0.0, 0.5])
    eta = np.array([0.25, 0.5, 0.75])
    node_coords = np.array([[-1.0, -1.0], [1.0, -0.5], [0.5, 1.0], [-1.5, 0.5], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    grad_phys = fcn(node_coords, node_values, xi, eta)
    grad_analytic = np.array([[beta] * len(xi), [gamma] * len(xi)])
    np.testing.assert_allclose(grad_phys, grad_analytic, rtol=1e-12, atol=1e-12)