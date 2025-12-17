def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """
    Test the identity isoparametric mapping (x=ξ, y=η). For a quadratic field
    u(ξ,η)=a+bξ+cη+dξ²+eξη+fη², the physical gradient matches the analytic gradient
    [∂u/∂x, ∂u/∂y] = [b + 2d ξ + e η, c + e ξ + 2f η] at multiple points.
    """
    import numpy as np
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    a, b, c, d, e, f = (0.3, -0.7, 1.1, 0.25, -0.4, 0.6)
    xi_nodes = node_coords[:, 0]
    eta_nodes = node_coords[:, 1]
    node_values = a + b * xi_nodes + c * eta_nodes + d * xi_nodes ** 2 + e * xi_nodes * eta_nodes + f * eta_nodes ** 2
    xi = np.array([-0.8, -0.3, 0.0, 0.35, 0.9, -0.5, 0.7], dtype=float)
    eta = np.array([-0.9, 0.6, 0.0, -0.25, 0.8, 0.5, -0.7], dtype=float)
    expected = np.vstack([b + 2.0 * d * xi + e * eta, c + e * xi + 2.0 * f * eta])
    grad = fcn(node_coords, node_values, xi, eta)
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (2, xi.size)
    assert np.allclose(grad, expected, rtol=1e-12, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """
    Test that for an isoparametric Q8 element under a nontrivial (curved or distorted) mapping,
    a linear physical field u(x,y)=α+βx+γy is reproduced exactly. The computed physical gradient
    should be constant [β, γ]^T at all evaluation points.
    """
    import numpy as np
    c1 = np.array([0.0, 0.0])
    c2 = np.array([2.0, 0.5])
    c3 = np.array([2.3, 2.0])
    c4 = np.array([-0.3, 1.7])
    n5 = 0.5 * (c1 + c2)
    n6 = 0.5 * (c2 + c3)
    n7 = 0.5 * (c3 + c4)
    n8 = 0.5 * (c4 + c1)
    node_coords = np.vstack([c1, c2, c3, c4, n5, n6, n7, n8])
    alpha, beta, gamma = (1.5, -0.8, 0.3)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    xi = np.array([-0.75, -0.2, 0.0, 0.4, 0.8, 0.6], dtype=float)
    eta = np.array([-0.6, 0.4, -0.1, 0.5, -0.3, 0.7], dtype=float)
    grad = fcn(node_coords, node_values, xi, eta)
    expected = np.vstack([np.full(xi.size, beta), np.full(xi.size, gamma)])
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (2, xi.size)
    assert np.allclose(grad, expected, rtol=1e-12, atol=1e-12)