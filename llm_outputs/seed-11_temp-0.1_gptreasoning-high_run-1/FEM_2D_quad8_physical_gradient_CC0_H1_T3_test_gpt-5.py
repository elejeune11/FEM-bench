def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """
    Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient.
    """
    import numpy as np
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    a, b, c, d, e, f = (0.7, -1.3, 2.1, 0.5, -0.9, 1.2)

    def u_val(xi, eta):
        return a + b * xi + c * eta + d * xi ** 2 + e * xi * eta + f * eta ** 2
    xi_nodes = node_coords[:, 0]
    eta_nodes = node_coords[:, 1]
    node_values = u_val(xi_nodes, eta_nodes)
    xi = np.array([-0.75, -0.3, 0.0, 0.42, 0.85], dtype=float)
    eta = np.array([-0.65, 0.2, -0.1, 0.51, 0.77], dtype=float)
    grad = fcn(node_coords, node_values, xi, eta)
    n = xi.size
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (2, n)
    grad_x = b + 2.0 * d * xi + e * eta
    grad_y = c + e * xi + 2.0 * f * eta
    expected = np.vstack([grad_x, grad_y])
    assert np.allclose(grad, expected, rtol=1e-12, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """
    Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points.
    """
    import numpy as np
    node_coords = np.array([[0.0, 0.0], [2.0, 0.1], [2.1, 1.8], [-0.2, 1.1], [1.0, -0.2], [2.2, 0.7], [1.0, 1.7], [-0.1, 0.6]], dtype=float)
    alpha, beta, gamma = (0.4, -2.0, 1.5)
    x_nodes = node_coords[:, 0]
    y_nodes = node_coords[:, 1]
    node_values = alpha + beta * x_nodes + gamma * y_nodes
    xi = np.array([-0.5, -0.1, 0.3, 0.4, 0.0], dtype=float)
    eta = np.array([-0.4, 0.2, 0.1, 0.45, -0.2], dtype=float)
    grad = fcn(node_coords, node_values, xi, eta)
    n = xi.size
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (2, n)
    expected = np.vstack([np.full(n, beta), np.full(n, gamma)])
    assert np.allclose(grad, expected, rtol=1e-12, atol=1e-12)