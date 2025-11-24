def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """
    Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient.
    """
    (a, b, c, d, e, f) = (0.7, -1.2, 2.3, 0.5, -0.9, 1.1)

    def u(xi, eta):
        return a + b * xi + c * eta + d * xi ** 2 + e * xi * eta + f * eta ** 2

    def grad_u(xi, eta):
        du_dxi = b + 2 * d * xi + e * eta
        du_deta = c + e * xi + 2 * f * eta
        return (du_dxi, du_deta)
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    node_values = np.array([u(xi, eta) for (xi, eta) in node_coords], dtype=float)
    xi = np.array([-1.0, -0.4, 0.0, 0.3, 0.9], dtype=float)
    eta = np.array([-1.0, 0.2, -0.5, 0.7, 1.0], dtype=float)
    grad = fcn(node_coords, node_values, xi, eta)
    (du_dxi, du_deta) = grad_u(xi, eta)
    expected = np.vstack([du_dxi, du_deta])
    assert grad.shape == expected.shape
    assert np.allclose(grad, expected, rtol=1e-12, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """
    Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points.
    """
    (alpha, beta, gamma) = (1.1, -2.0, 0.7)
    node_coords = np.array([[0.0, 0.0], [2.0, 0.2], [2.1, 1.9], [-0.1, 1.8], [1.0, -0.3], [2.4, 1.0], [1.0, 2.2], [-0.4, 1.0]], dtype=float)
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    node_values = alpha + beta * x + gamma * y
    xi = np.array([-0.6, -0.2, 0.0, 0.5, 0.9, 0.1], dtype=float)
    eta = np.array([-0.7, 0.0, 0.6, -0.4, 0.8, 0.2], dtype=float)
    grad = fcn(node_coords, node_values, xi, eta)
    expected = np.vstack([np.full_like(xi, beta, dtype=float), np.full_like(eta, gamma, dtype=float)])
    assert grad.shape == expected.shape
    assert np.allclose(grad, expected, rtol=1e-12, atol=1e-12)