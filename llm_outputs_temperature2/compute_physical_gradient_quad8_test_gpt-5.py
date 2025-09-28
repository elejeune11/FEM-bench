def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η). For a quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη², the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (a, b, c, d, e, f) = (0.3, -1.1, 0.7, 0.4, -0.8, 1.2)

    def u(xi, eta):
        return a + b * xi + c * eta + d * xi ** 2 + e * xi * eta + f * eta ** 2
    node_values = np.array([u(xi, eta) for (xi, eta) in node_coords], dtype=float)
    xi_eval = np.array([-0.6, 0.0, 0.9, -0.2], dtype=float)
    eta_eval = np.array([0.3, -0.5, 0.1, 0.8], dtype=float)
    grad_x = b + 2 * d * xi_eval + e * eta_eval
    grad_y = c + e * xi_eval + 2 * f * eta_eval
    expected = np.vstack([grad_x, grad_y])
    grad = fcn(node_coords, node_values, xi_eval, eta_eval)
    assert grad.shape == expected.shape
    assert np.allclose(grad, expected, rtol=1e-10, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field u(x,y)=α+βx+γy is reproduced exactly. The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[0.0, 0.0], [2.0, -0.2], [2.2, 1.3], [-0.1, 1.1], [1.1, -0.4], [2.4, 0.6], [1.0, 1.5], [-0.3, 0.5]], dtype=float)
    (alpha, beta, gamma) = (3.2, -0.75, 1.4)

    def u_phys(x, y):
        return alpha + beta * x + gamma * y
    node_values = np.array([u_phys(x, y) for (x, y) in node_coords], dtype=float)
    xi_eval = np.array([-0.8, -0.3, 0.0, 0.6, 0.9], dtype=float)
    eta_eval = np.array([-0.6, 0.2, 0.0, 0.7, -0.4], dtype=float)
    grad = fcn(node_coords, node_values, xi_eval, eta_eval)
    expected = np.vstack([np.full_like(xi_eval, beta, dtype=float), np.full_like(xi_eval, gamma, dtype=float)])
    assert grad.shape == expected.shape
    assert np.allclose(grad, expected, rtol=1e-10, atol=1e-12)