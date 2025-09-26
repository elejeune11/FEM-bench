def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (a, b, c, d, e, f) = (1.5, -2.1, 3.2, 0.5, 1.8, -0.9)

    def u(xi, eta):
        return a + b * xi + c * eta + d * xi ** 2 + e * xi * eta + f * eta ** 2
    node_values = u(node_coords[:, 0], node_coords[:, 1])
    xi_pts = np.array([0.0, 0.5, -0.25, 0.1, -0.9, 1.0, -1.0])
    eta_pts = np.array([0.0, -0.5, 0.75, 0.8, 0.3, 1.0, -1.0])
    grad_x_analytic = b + 2 * d * xi_pts + e * eta_pts
    grad_y_analytic = c + e * xi_pts + 2 * f * eta_pts
    expected_grad = np.vstack([grad_x_analytic, grad_y_analytic])
    computed_grad = fcn(node_coords, node_values, xi_pts, eta_pts)
    assert computed_grad.shape == expected_grad.shape
    assert np.allclose(computed_grad, expected_grad)
    (xi_scalar, eta_scalar) = (0.2, -0.3)
    grad_x_scalar = b + 2 * d * xi_scalar + e * eta_scalar
    grad_y_scalar = c + e * xi_scalar + 2 * f * eta_scalar
    expected_scalar = np.array([[grad_x_scalar], [grad_y_scalar]])
    computed_scalar = fcn(node_coords, node_values, xi_scalar, eta_scalar)
    assert computed_scalar.shape == (2, 1)
    assert np.allclose(computed_scalar, expected_scalar)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[0.0, 0.0], [3.0, 0.5], [2.5, 2.5], [0.2, 2.0], [1.5, -0.1], [3.1, 1.5], [1.2, 2.6], [-0.2, 1.0]])
    (alpha, beta, gamma) = (5.0, 2.0, -3.0)

    def u(x, y):
        return alpha + beta * x + gamma * y
    node_values = u(node_coords[:, 0], node_coords[:, 1])
    xi_pts = np.array([0.0, 0.5, -0.25, 0.1, -0.9, 1.0, -1.0])
    eta_pts = np.array([0.0, -0.5, 0.75, 0.8, 0.3, 1.0, -1.0])
    n_pts = len(xi_pts)
    expected_grad = np.full((2, n_pts), [[beta], [gamma]])
    computed_grad = fcn(node_coords, node_values, xi_pts, eta_pts)
    assert computed_grad.shape == expected_grad.shape
    assert np.allclose(computed_grad, expected_grad)
    (xi_scalar, eta_scalar) = (-0.7, 0.6)
    expected_scalar = np.array([[beta], [gamma]])
    computed_scalar = fcn(node_coords, node_values, xi_scalar, eta_scalar)
    assert computed_scalar.shape == (2, 1)
    assert np.allclose(computed_scalar, expected_scalar)