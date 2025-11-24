def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (a, b, c, d, e, f) = (1.0, -2.5, 3.0, 0.5, -1.5, 2.0)

    def u(xi, eta):
        return a + b * xi + c * eta + d * xi ** 2 + e * xi * eta + f * eta ** 2
    node_xi = node_coords[:, 0]
    node_eta = node_coords[:, 1]
    node_values = u(node_xi, node_eta)
    xi_pts = np.array([-0.9, -0.5, 0.0, 0.5, 0.9, 0.25])
    eta_pts = np.array([0.8, -0.4, 0.0, 0.6, -0.7, -0.15])
    grad_computed = fcn(node_coords, node_values, xi_pts, eta_pts)
    grad_analytic_x = b + 2 * d * xi_pts + e * eta_pts
    grad_analytic_y = c + e * xi_pts + 2 * f * eta_pts
    grad_analytic = np.vstack([grad_analytic_x, grad_analytic_y])
    assert np.allclose(grad_computed, grad_analytic)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.1, -1.2], [1.2, 0.1], [-0.1, 1.2], [-1.2, -0.1]])
    (alpha, beta, gamma) = (10.0, -5.0, 2.5)

    def u(x, y):
        return alpha + beta * x + gamma * y
    node_x = node_coords[:, 0]
    node_y = node_coords[:, 1]
    node_values = u(node_x, node_y)
    xi_pts = np.array([-0.9, -0.5, 0.0, 0.5, 0.9, 0.25])
    eta_pts = np.array([0.8, -0.4, 0.0, 0.6, -0.7, -0.15])
    n_pts = len(xi_pts)
    grad_computed = fcn(node_coords, node_values, xi_pts, eta_pts)
    expected_grad = np.array([[beta] * n_pts, [gamma] * n_pts])
    assert np.allclose(grad_computed, expected_grad)