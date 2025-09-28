def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (a, b, c, d, e, f) = (2, 3, 4, 5, 6, 7)
    node_values = np.zeros(8)
    for (i, (xi_node, eta_node)) in enumerate(node_coords):
        node_values[i] = a + b * xi_node + c * eta_node + d * xi_node ** 2 + e * xi_node * eta_node + f * eta_node ** 2
    xi_test = np.array([0.0, -0.5, 0.5, 0.25, -0.25])
    eta_test = np.array([0.0, 0.5, -0.5, 0.75, -0.75])
    grad_phys = fcn(node_coords, node_values, xi_test, eta_test)
    expected_grad_x = b + 2 * d * xi_test + e * eta_test
    expected_grad_y = c + e * xi_test + 2 * f * eta_test
    assert np.allclose(grad_phys[0], expected_grad_x, rtol=1e-10, atol=1e-12)
    assert np.allclose(grad_phys[1], expected_grad_y, rtol=1e-10, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[0.0, 0.0], [2.0, 0.1], [2.2, 2.0], [0.1, 1.9], [1.0, 0.05], [2.1, 1.05], [1.15, 1.95], [0.05, 0.95]])
    (alpha, beta, gamma) = (5.0, 3.0, 7.0)
    node_values = np.zeros(8)
    for (i, (x, y)) in enumerate(node_coords):
        node_values[i] = alpha + beta * x + gamma * y
    xi_test = np.array([0.0, -0.3, 0.7, -0.8, 0.4, 0.9, -0.6])
    eta_test = np.array([0.0, 0.5, -0.2, 0.8, -0.9, 0.3, -0.4])
    grad_phys = fcn(node_coords, node_values, xi_test, eta_test)
    expected_grad_x = np.full(len(xi_test), beta)
    expected_grad_y = np.full(len(eta_test), gamma)
    assert np.allclose(grad_phys[0], expected_grad_x, rtol=1e-10, atol=1e-12)
    assert np.allclose(grad_phys[1], expected_grad_y, rtol=1e-10, atol=1e-12)