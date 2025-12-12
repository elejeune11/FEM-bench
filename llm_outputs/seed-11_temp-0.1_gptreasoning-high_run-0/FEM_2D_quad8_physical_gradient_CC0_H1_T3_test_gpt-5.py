def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """
    Test the identity isoparametric mapping (x=ξ, y=η). For a quadratic field
    u(ξ,η)=a+bξ+cη+dξ²+eξη+fη², the physical gradient equals the analytic gradient.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    (a, b, c, d, e, f) = (-0.3, 1.2, -0.7, 0.5, -0.25, 0.8)

    def u_quad(xi, eta):
        return a + b * xi + c * eta + d * xi ** 2 + e * xi * eta + f * eta ** 2
    xi_nodes = node_coords[:, 0]
    eta_nodes = node_coords[:, 1]
    node_values = u_quad(xi_nodes, eta_nodes)
    xi_eval = np.array([-0.75, -0.1, 0.0, 0.45, 0.7, -0.3], dtype=float)
    eta_eval = np.array([-0.55, 0.2, -0.4, 0.6, -0.1, 0.9], dtype=float)
    grad = fcn(node_coords, node_values, xi_eval, eta_eval)
    du_dxi = b + 2.0 * d * xi_eval + e * eta_eval
    du_deta = c + e * xi_eval + 2.0 * f * eta_eval
    expected = np.vstack((du_dxi, du_deta))
    assert grad.shape == expected.shape
    assert np.allclose(grad, expected, rtol=1e-12, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """
    Test that under any isoparametric mapping, a linear physical field u(x,y)=α+βx+γy
    is reproduced exactly by quadratic quadrilateral elements. The physical gradient
    should be [β, γ]^T at all points.
    """
    node_coords = np.array([[0.0, 0.0], [2.0, 0.1], [2.1, 2.0], [0.1, 1.9], [1.0, -0.15], [2.2, 1.05], [1.1, 2.05], [-0.05, 0.95]], dtype=float)
    (alpha, beta, gamma) = (0.7, -1.3, 2.2)
    x_nodes = node_coords[:, 0]
    y_nodes = node_coords[:, 1]
    node_values = alpha + beta * x_nodes + gamma * y_nodes
    xi_eval = np.array([-0.8, -0.4, 0.0, 0.3, 0.7, 0.9, -0.1], dtype=float)
    eta_eval = np.array([-0.6, 0.5, -0.2, 0.1, 0.8, -0.7, 0.0], dtype=float)
    grad = fcn(node_coords, node_values, xi_eval, eta_eval)
    n = xi_eval.size
    expected = np.vstack((beta * np.ones(n), gamma * np.ones(n)))
    assert grad.shape == expected.shape
    assert np.allclose(grad, expected, rtol=1e-12, atol=1e-12)