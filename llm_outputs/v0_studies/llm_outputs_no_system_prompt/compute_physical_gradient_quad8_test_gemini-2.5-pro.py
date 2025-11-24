def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (a, b, c, d, e, f) = (1.0, 2.0, -3.0, 0.5, -1.5, 2.5)

    def u(xi, eta):
        return a + b * xi + c * eta + d * xi ** 2 + e * xi * eta + f * eta ** 2
    node_xi = node_coords[:, 0]
    node_eta = node_coords[:, 1]
    node_values = u(node_xi, node_eta)

    def grad_u_analytic(xi, eta):
        dudxi = b + 2 * d * xi + e * eta
        dudeta = c + e * xi + 2 * f * eta
        return np.vstack([dudxi, dudeta])
    xi = np.array([0.0, 0.5, -0.25, 0.9, -0.8])
    eta = np.array([0.0, -0.5, 0.75, -0.1, 0.2])
    grad_phys_computed = fcn(node_coords, node_values, xi, eta)
    grad_phys_expected = grad_u_analytic(xi, eta)
    np.testing.assert_allclose(grad_phys_computed, grad_phys_expected)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.2, 1.1], [-1.0, 0.0]])
    (alpha, beta, gamma) = (5.0, -2.0, 3.5)

    def u(x, y):
        return alpha + beta * x + gamma * y
    node_x = node_coords[:, 0]
    node_y = node_coords[:, 1]
    node_values = u(node_x, node_y)
    xi = np.array([0.0, 0.5, -0.25, 0.9, -0.8, 0.123])
    eta = np.array([0.0, -0.5, 0.75, -0.1, 0.2, -0.456])
    n_pts = len(xi)
    grad_phys_computed = fcn(node_coords, node_values, xi, eta)
    grad_phys_expected = np.full((2, n_pts), [[beta], [gamma]])
    np.testing.assert_allclose(grad_phys_computed, grad_phys_expected)