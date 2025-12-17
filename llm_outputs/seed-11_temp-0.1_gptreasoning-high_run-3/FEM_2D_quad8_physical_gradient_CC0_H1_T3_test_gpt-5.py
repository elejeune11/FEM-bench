def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """
    Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    a, b, c, d, e, f = (0.7, -1.3, 2.0, 0.5, -0.6, 1.2)
    xi_nodes = node_coords[:, 0]
    eta_nodes = node_coords[:, 1]
    node_values = a + b * xi_nodes + c * eta_nodes + d * xi_nodes ** 2 + e * xi_nodes * eta_nodes + f * eta_nodes ** 2
    xi_eval = np.array([-0.93, -0.5, 0.0, 0.35, 0.73, 0.99])
    eta_eval = np.array([-0.6, 0.2, -0.1, 0.87, -0.77, 0.4])
    grad = fcn(node_coords, node_values, xi_eval, eta_eval)
    dudx = b + 2.0 * d * xi_eval + e * eta_eval
    dudy = c + e * xi_eval + 2.0 * f * eta_eval
    expected = np.vstack([dudx, dudy])
    assert grad.shape == expected.shape
    assert np.allclose(grad, expected, rtol=1e-12, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """
    Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points.
    """
    node_coords = np.array([[0.0, 0.0], [2.2, -0.2], [2.5, 1.9], [0.1, 2.3], [1.2, -0.6], [2.8, 0.9], [1.3, 2.7], [-0.4, 1.0]])
    alpha, beta, gamma = (-0.25, 1.7, -0.9)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    xi_eval = np.array([-0.8, -0.6, -0.2, 0.0, 0.25, 0.5, 0.7])
    eta_eval = np.array([-0.7, -0.1, 0.3, 0.6, -0.5, 0.2, 0.75])
    grad = fcn(node_coords, node_values, xi_eval, eta_eval)
    expected = np.vstack([beta * np.ones_like(xi_eval), gamma * np.ones_like(xi_eval)])
    assert grad.shape == expected.shape
    assert np.allclose(grad, expected, rtol=1e-12, atol=1e-12)