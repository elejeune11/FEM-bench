def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (a, b, c, d, e, f) = (2.0, 3.0, -1.5, 0.5, 2.0, -0.8)
    node_values = np.zeros(8)
    for (i, (xi_node, eta_node)) in enumerate(node_coords):
        node_values[i] = a + b * xi_node + c * eta_node + d * xi_node ** 2 + e * xi_node * eta_node + f * eta_node ** 2
    xi_test = np.array([-0.5, 0.3, 0.8])
    eta_test = np.array([0.2, -0.7, 0.4])
    grad_phys = fcn(node_coords, node_values, xi_test, eta_test)
    grad_analytic = np.zeros((2, len(xi_test)))
    for i in range(len(xi_test)):
        grad_analytic[0, i] = b + 2 * d * xi_test[i] + e * eta_test[i]
        grad_analytic[1, i] = c + e * xi_test[i] + 2 * f * eta_test[i]
    assert np.allclose(grad_phys, grad_analytic, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[-1.2, -0.8], [1.1, -1.1], [0.9, 1.2], [-1.1, 0.9], [0.1, -0.95], [1.05, 0.05], [-0.05, 1.05], [-1.05, -0.1]])
    (alpha, beta, gamma) = (5.0, 2.0, -3.0)
    node_values = np.zeros(8)
    for i in range(8):
        (x, y) = node_coords[i]
        node_values[i] = alpha + beta * x + gamma * y
    xi_test = np.array([-0.6, 0.0, 0.4, 0.7])
    eta_test = np.array([0.3, -0.2, 0.8, -0.5])
    grad_phys = fcn(node_coords, node_values, xi_test, eta_test)
    expected_grad = np.array([[beta], [gamma]])
    expected_grad = np.tile(expected_grad, (1, len(xi_test)))
    assert np.allclose(grad_phys, expected_grad, atol=1e-12)