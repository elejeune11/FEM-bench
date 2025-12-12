def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """
    Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient.
    """
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (a, b, c, d, e, f) = (2, 3, 4, 5, 6, 7)
    node_values = np.array([a - b - c + d + e + f, a + b - c + d - e - f, a + b + c + d + e + f, a - b + c + d - e + f, a - c + d - f, a + b + d + f, a + c + d + f, a - b + d - f])
    xi = np.array([0.0, 0.5, -0.5])
    eta = np.array([0.0, 0.3, -0.7])
    grad_analytic = np.array([[b + 10 * xi + 6 * eta], [c + 6 * xi + 14 * eta]])
    grad_phys = fcn(node_coords, node_values, xi, eta)
    np.testing.assert_allclose(grad_phys, grad_analytic, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """
    Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points.
    """
    node_coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, -0.1], [2.1, 1.0], [1.0, 2.1], [-0.1, 1.0]])
    (alpha, beta, gamma) = (3, 4, 5)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    xi = np.array([0.0, 0.5, -0.5])
    eta = np.array([0.0, 0.3, -0.7])
    grad_expected = np.array([[beta] * len(xi), [gamma] * len(xi)])
    grad_phys = fcn(node_coords, node_values, xi, eta)
    np.testing.assert_allclose(grad_phys, grad_expected, atol=1e-12)