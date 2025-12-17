def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """
    Test the identity isoparametric mapping (x=ξ, y=η). For a quadratic field
    u(ξ,η) = a + bξ + cη + dξ² + eξη + fη², the physical gradient should
    equal the analytic gradient [∂u/∂x, ∂u/∂y] at multiple points.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    a = 0.7
    b = -1.2
    c = 0.4
    d = 0.3
    e = -0.5
    f = 0.9
    xi_nodes = node_coords[:, 0]
    eta_nodes = node_coords[:, 1]
    node_values = a + b * xi_nodes + c * eta_nodes + d * xi_nodes ** 2 + e * xi_nodes * eta_nodes + f * eta_nodes ** 2
    xi = np.array([-0.8, -0.4, 0.0, 0.3, 0.9])
    eta = np.array([0.75, -0.2, 0.0, -0.7, 0.2])
    n = xi.size
    grad = fcn(node_coords, node_values, xi, eta)
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (2, n)
    du_dx = b + 2.0 * d * xi + e * eta
    du_dy = c + e * xi + 2.0 * f * eta
    grad_expected = np.vstack([du_dx, du_dy])
    assert np.allclose(grad, grad_expected, rtol=1e-12, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """
    Under any isoparametric (possibly curved) mapping, a linear physical field
    u(x,y) = α + βx + γy is reproduced exactly by Q8 elements. The physical
    gradient should be constant [β, γ]^T at all evaluation points.
    """
    node_coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.5], [0.0, 1.5], [1.0, -0.3], [2.2, 0.75], [1.0, 1.8], [-0.2, 0.75]])
    alpha = -0.6
    beta = 1.3
    gamma = -0.8
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    node_values = alpha + beta * x + gamma * y
    xi = np.array([-0.75, -0.2, 0.0, 0.4, 0.8, 0.3])
    eta = np.array([-0.6, 0.5, -0.1, 0.7, -0.4, 0.2])
    n = xi.size
    grad = fcn(node_coords, node_values, xi, eta)
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (2, n)
    grad_expected = np.vstack([np.full(n, beta), np.full(n, gamma)])
    assert np.allclose(grad, grad_expected, rtol=1e-12, atol=1e-12)