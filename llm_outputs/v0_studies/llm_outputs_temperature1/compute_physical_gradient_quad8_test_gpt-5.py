def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """
    Test the identity isoparametric mapping (x=ξ, y=η). For a quadratic field
    u(ξ,η)=a+bξ+cη+dξ²+eξη+fη², the physical gradient equals the analytic gradient.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (a, b, c, d, e, f) = (0.7, -1.2, 0.3, 0.5, -0.25, 1.1)
    xi_nodes = node_coords[:, 0]
    eta_nodes = node_coords[:, 1]
    node_values = a + b * xi_nodes + c * eta_nodes + d * xi_nodes ** 2 + e * xi_nodes * eta_nodes + f * eta_nodes ** 2
    xi = np.array([-1.0, -0.3, 0.0, 0.7, 1.0])
    eta = np.array([-1.0, 0.4, 0.0, -0.2, 1.0])
    grad = fcn(node_coords, node_values, xi, eta)
    expected = np.vstack([b + 2.0 * d * xi + e * eta, c + e * xi + 2.0 * f * eta])
    assert grad.shape == (2, xi.size)
    assert np.allclose(grad, expected, rtol=1e-12, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """
    Test that for any isoparametric mapping, a linear physical field u(x,y)=α+βx+γy
    is reproduced exactly by Q8 elements. The physical gradient should be [β, γ]^T at all points.
    """
    node_coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0], [1.0, -0.2], [2.2, 0.5], [1.0, 1.2], [-0.2, 0.5]])
    (alpha, beta, gamma) = (1.1, -0.4, 2.3)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    xi = np.array([-1.0, -0.6, -0.1, 0.0, 0.4, 0.8, 1.0])
    eta = np.array([-1.0, -0.2, 0.3, 0.9, -0.5, 0.1, 1.0])
    grad = fcn(node_coords, node_values, xi, eta)
    expected = np.tile(np.array([[beta], [gamma]]), (1, xi.size))
    assert grad.shape == (2, xi.size)
    assert np.allclose(grad, expected, rtol=1e-12, atol=1e-12)