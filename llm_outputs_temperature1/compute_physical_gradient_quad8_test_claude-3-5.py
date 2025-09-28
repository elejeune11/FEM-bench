def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη², 
    the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (a, b, c, d, e, f) = (1.0, 2.0, -1.0, 0.5, -0.3, 0.7)
    node_values = np.array([a + b * x + c * y + d * x ** 2 + e * x * y + f * y ** 2 for (x, y) in node_coords])
    xi = np.array([-0.5, 0.0, 0.25])
    eta = np.array([0.1, -0.3, 0.7])
    grad = fcn(node_coords, node_values, xi, eta)
    grad_x = b + 2 * d * xi + e * eta
    grad_y = c + e * xi + 2 * f * eta
    grad_true = np.vstack([grad_x, grad_y])
    assert np.allclose(grad, grad_true)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[-1, -1], [1, -1], [1.2, 1.1], [-0.8, 0.9], [0, -1.1], [1.1, 0], [0.2, 1], [-0.9, 0]])
    (alpha, beta, gamma) = (1.0, -0.5, 2.0)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    xi = np.array([-0.7, 0.1, 0.4])
    eta = np.array([0.2, -0.5, 0.3])
    grad = fcn(node_coords, node_values, xi, eta)
    grad_true = np.array([[beta, beta, beta], [gamma, gamma, gamma]])
    assert np.allclose(grad, grad_true)