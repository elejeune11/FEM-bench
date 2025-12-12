def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient.
    """
    import numpy as np
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (a, b, c, d, e, f) = (0.7, -1.3, 2.1, 0.5, -0.8, 1.2)
    xi_nodes = node_coords[:, 0]
    eta_nodes = node_coords[:, 1]
    node_values = a + b * xi_nodes + c * eta_nodes + d * xi_nodes ** 2 + e * xi_nodes * eta_nodes + f * eta_nodes ** 2
    xi = np.array([-0.9, -0.3, 0.0, 0.45, 0.9])
    eta = np.array([-0.6, 0.1, 0.0, 0.7, 0.95])
    grad = fcn(node_coords, node_values, xi, eta)
    grad_analytic = np.vstack((b + 2.0 * d * xi + e * eta, c + e * xi + 2.0 * f * eta))
    assert grad.shape == (2, xi.size)
    assert np.allclose(grad, grad_analytic, atol=1e-09)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points.
    """
    import numpy as np
    node_coords = np.array([[0.0, 0.0], [2.0, 0.1], [2.2, 1.1], [-0.1, 1.0], [1.0, -0.3], [2.3, 0.5], [1.05, 1.4], [-0.25, 0.45]])
    alpha = 0.4
    beta = -1.7
    gamma = 2.3
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    xi = np.linspace(-0.8, 0.8, 6)
    eta = np.array([-0.9, -0.2, 0.0, 0.3, 0.6, 0.9])
    grad = fcn(node_coords, node_values, xi, eta)
    expected = np.vstack((np.full(xi.size, beta), np.full(xi.size, gamma)))
    assert grad.shape == (2, xi.size)
    assert np.allclose(grad, expected, atol=1e-09)