def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """
    Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (a, b, c, d, e, f_coeff) = (2.0, 0.5, -1.2, 0.3, 0.8, -0.4)
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    node_values = a + b * x + c * y + d * x ** 2 + e * x * y + f_coeff * y ** 2
    xi = np.array([0.0, 0.5, -0.3, 0.8])
    eta = np.array([0.0, -0.5, 0.2, 0.9])
    grad_phys = fcn(node_coords, node_values, xi, eta)
    du_dx = b + 2 * d * xi + e * eta
    du_dy = c + e * xi + 2 * f_coeff * eta
    expected_grad = np.vstack((du_dx, du_dy))
    np.testing.assert_allclose(grad_phys, expected_grad, atol=1e-14, rtol=1e-14)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """
    Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points.
    """
    corners = np.array([[-2.0, -1.5], [3.0, -1.0], [2.5, 3.0], [-1.5, 2.0]])
    mids = np.array([[0.5, -1.4], [2.8, 1.0], [0.5, 2.6], [-1.8, 0.2]])
    node_coords = np.vstack((corners, mids))
    (alpha, beta, gamma) = (10.0, -2.0, 3.0)
    x_nodes = node_coords[:, 0]
    y_nodes = node_coords[:, 1]
    node_values = alpha + beta * x_nodes + gamma * y_nodes
    xi = np.array([0.0, -0.5, 0.5, 0.1])
    eta = np.array([0.0, 0.5, -0.5, -0.2])
    grad_phys = fcn(node_coords, node_values, xi, eta)
    expected_val = np.array([[beta], [gamma]])
    expected_grad = np.tile(expected_val, (1, len(xi)))
    np.testing.assert_allclose(grad_phys, expected_grad, atol=1e-13, rtol=1e-13)