def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (a, b, c, d, e, f) = (1.5, -2.0, 0.5, 1.2, -0.8, 2.1)

    def u_field(x, y):
        return a + b * x + c * y + d * x ** 2 + e * x * y + f * y ** 2
    node_values = u_field(node_coords[:, 0], node_coords[:, 1])

    def analytic_grad(x, y):
        grad_x = b + 2 * d * x + e * y
        grad_y = c + e * x + 2 * f * y
        return np.vstack([grad_x, grad_y])
    xi = np.array([0.0, 0.5, -0.25, 0.9, -1.0])
    eta = np.array([0.0, -0.5, 0.75, 0.8, 1.0])
    expected_grad = analytic_grad(xi, eta)
    computed_grad = fcn(node_coords, node_values, xi, eta)
    assert computed_grad.shape == expected_grad.shape
    assert np.allclose(computed_grad, expected_grad)
    (xi_s, eta_s) = (0.1, -0.3)
    expected_grad_s = analytic_grad(xi_s, eta_s)
    computed_grad_s = fcn(node_coords, node_values, xi_s, eta_s)
    assert np.allclose(computed_grad_s, expected_grad_s)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[0.0, 0.0], [2.0, 0.5], [2.5, 2.5], [0.5, 2.0], [1.0, -0.2], [2.4, 1.5], [1.3, 2.6], [-0.1, 1.0]])
    (alpha, beta, gamma) = (10.0, -3.5, 2.8)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    expected_grad_vector = np.array([beta, gamma])
    xi = np.array([0.0, 0.5, -0.25, 0.9, -1.0, 1.0])
    eta = np.array([0.0, -0.5, 0.75, 0.8, 1.0, -1.0])
    n_pts = len(xi)
    expected_grad = np.tile(expected_grad_vector.reshape(-1, 1), (1, n_pts))
    computed_grad = fcn(node_coords, node_values, xi, eta)
    assert computed_grad.shape == expected_grad.shape
    assert np.allclose(computed_grad, expected_grad)
    (xi_s, eta_s) = (0.1, -0.3)
    expected_grad_s = expected_grad_vector.reshape(-1, 1)
    computed_grad_s = fcn(node_coords, node_values, xi_s, eta_s)
    assert np.allclose(computed_grad_s, expected_grad_s)