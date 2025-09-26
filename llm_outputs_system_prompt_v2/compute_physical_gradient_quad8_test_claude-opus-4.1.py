def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (a, b, c, d, e, f) = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    node_values = np.zeros(8)
    for i in range(8):
        (xi_i, eta_i) = node_coords[i]
        node_values[i] = a + b * xi_i + c * eta_i + d * xi_i ** 2 + e * xi_i * eta_i + f * eta_i ** 2
    test_xi = np.array([0.0, -0.5, 0.5, 0.25, -0.75])
    test_eta = np.array([0.0, 0.5, -0.5, 0.75, -0.25])
    grad_phys = fcn(node_coords, node_values, test_xi, test_eta)
    for j in range(len(test_xi)):
        (xi_j, eta_j) = (test_xi[j], test_eta[j])
        expected_grad_xi = b + 2 * d * xi_j + e * eta_j
        expected_grad_eta = c + e * xi_j + 2 * f * eta_j
        assert np.abs(grad_phys[0, j] - expected_grad_xi) < 1e-10
        assert np.abs(grad_phys[1, j] - expected_grad_eta) < 1e-10

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.5, 2.0], [0.0, 2.0], [1.0, -0.1], [2.3, 1.0], [1.25, 2.1], [-0.1, 1.0]])
    (alpha, beta, gamma) = (3.0, 5.0, 7.0)
    node_values = np.zeros(8)
    for i in range(8):
        (x_i, y_i) = node_coords[i]
        node_values[i] = alpha + beta * x_i + gamma * y_i
    test_xi = np.array([0.0, -0.5, 0.5, 0.25, -0.75, 0.8, -0.3])
    test_eta = np.array([0.0, 0.5, -0.5, 0.75, -0.25, 0.6, 0.4])
    grad_phys = fcn(node_coords, node_values, test_xi, test_eta)
    expected_grad_x = beta
    expected_grad_y = gamma
    for j in range(len(test_xi)):
        assert np.abs(grad_phys[0, j] - expected_grad_x) < 1e-10
        assert np.abs(grad_phys[1, j] - expected_grad_y) < 1e-10