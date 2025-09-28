def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (a, b, c, d, e, f) = (1.0, 2.5, -3.1, 0.8, -1.2, 2.0)

    def u_analytic(x, y):
        return a + b * x + c * y + d * x ** 2 + e * x * y + f * y ** 2

    def grad_u_analytic(x, y):
        dudx = b + 2 * d * x + e * y
        dudy = c + e * x + 2 * f * y
        return np.vstack([dudx, dudy])
    node_xi = node_coords[:, 0]
    node_eta = node_coords[:, 1]
    node_values = u_analytic(node_xi, node_eta)
    (xi_pts, eta_pts) = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    xi = xi_pts.flatten()
    eta = eta_pts.flatten()
    grad_computed = fcn(node_coords, node_values, xi, eta)
    grad_expected = grad_u_analytic(xi, eta)
    np.testing.assert_allclose(grad_computed, grad_expected)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.1, -1.2], [1.1, -0.1], [0.2, 1.1], [-1.2, 0.2]])
    (alpha, beta, gamma) = (5.0, -2.5, 3.8)

    def u_analytic(x, y):
        return alpha + beta * x + gamma * y
    node_x = node_coords[:, 0]
    node_y = node_coords[:, 1]
    node_values = u_analytic(node_x, node_y)
    (xi_pts, eta_pts) = np.meshgrid(np.linspace(-0.9, 0.9, 4), np.linspace(-0.9, 0.9, 4))
    xi = xi_pts.flatten()
    eta = eta_pts.flatten()
    n_pts = len(xi)
    grad_computed = fcn(node_coords, node_values, xi, eta)
    grad_expected = np.full((2, n_pts), [[beta], [gamma]])
    np.testing.assert_allclose(grad_computed, grad_expected)