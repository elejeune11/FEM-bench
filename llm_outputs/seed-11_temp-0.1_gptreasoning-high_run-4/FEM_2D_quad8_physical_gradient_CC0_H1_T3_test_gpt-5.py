def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """
    Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient at sampled points.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    a, b, c, d, e, f = (0.7, -0.3, 0.5, 0.2, -0.4, 0.6)

    def u_quad(xi, eta):
        return a + b * xi + c * eta + d * xi ** 2 + e * xi * eta + f * eta ** 2
    xi_nodes = node_coords[:, 0]
    eta_nodes = node_coords[:, 1]
    node_values = u_quad(xi_nodes, eta_nodes)
    xi_pts = np.array([-0.8, -0.2, 0.0, 0.3, 0.9], dtype=float)
    eta_pts = np.array([-0.9, 0.0, 0.4, -0.5, 0.7], dtype=float)
    grad_x_analytic = b + 2 * d * xi_pts + e * eta_pts
    grad_y_analytic = c + e * xi_pts + 2 * f * eta_pts
    grad = fcn(node_coords, node_values, xi_pts, eta_pts)
    assert grad.shape == (2, xi_pts.size)
    assert np.allclose(grad[0], grad_x_analytic, rtol=1e-12, atol=1e-12)
    assert np.allclose(grad[1], grad_y_analytic, rtol=1e-12, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """
    Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points.
    """
    nat_nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    xi_n = nat_nodes[:, 0]
    eta_n = nat_nodes[:, 1]

    def x_map(xi, eta):
        return xi + 0.2 * xi * eta + 0.1 * eta ** 2

    def y_map(xi, eta):
        return 1.2 * eta + 0.15 * xi ** 2 - 0.05 * xi * eta
    x_nodes = x_map(xi_n, eta_n)
    y_nodes = y_map(xi_n, eta_n)
    node_coords = np.column_stack([x_nodes, y_nodes])
    alpha, beta, gamma = (1.1, -0.4, 0.7)
    node_values = alpha + beta * x_nodes + gamma * y_nodes
    xi_pts = np.array([-0.6, -0.3, 0.0, 0.4, 0.6], dtype=float)
    eta_pts = np.array([-0.5, 0.4, 0.1, -0.4, 0.5], dtype=float)
    grad = fcn(node_coords, node_values, xi_pts, eta_pts)
    n = xi_pts.size
    expected = np.array([np.full(n, beta), np.full(n, gamma)])
    assert grad.shape == (2, n)
    assert np.allclose(grad, expected, rtol=1e-12, atol=1e-12)