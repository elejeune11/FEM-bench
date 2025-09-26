def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (a, b, c, d, e, f) = (1.0, -2.5, 3.0, 0.5, -1.5, 2.0)

    def u_field(xi, eta):
        return a + b * xi + c * eta + d * xi ** 2 + e * xi * eta + f * eta ** 2

    def grad_analytic_field(xi, eta):
        grad_xi = b + 2 * d * xi + e * eta
        grad_eta = c + e * xi + 2 * f * eta
        return np.array([grad_xi, grad_eta])
    node_values = u_field(node_coords[:, 0], node_coords[:, 1])
    (xi_grid, eta_grid) = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    xi_pts = xi_grid.ravel()
    eta_pts = eta_grid.ravel()
    grad_computed = fcn(node_coords, node_values, xi_pts, eta_pts)
    grad_analytic = grad_analytic_field(xi_pts, eta_pts)
    np.testing.assert_allclose(grad_computed, grad_analytic, atol=1e-14)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[-1.1, -0.9], [1.2, -1.1], [0.9, 1.2], [-0.8, 1.1], [0.1, -1.0], [1.0, -0.1], [0.2, 1.15], [-1.0, 0.2]])
    (alpha, beta, gamma) = (5.0, -2.0, 3.0)

    def u_field(x, y):
        return alpha + beta * x + gamma * y
    node_values = u_field(node_coords[:, 0], node_coords[:, 1])
    (xi_grid, eta_grid) = np.meshgrid(np.linspace(-0.95, 0.95, 4), np.linspace(-0.95, 0.95, 4))
    xi_pts = xi_grid.ravel()
    eta_pts = eta_grid.ravel()
    n_pts = len(xi_pts)
    grad_computed = fcn(node_coords, node_values, xi_pts, eta_pts)
    grad_analytic = np.full((2, n_pts), [[beta], [gamma]])
    np.testing.assert_allclose(grad_computed, grad_analytic, atol=1e-14)