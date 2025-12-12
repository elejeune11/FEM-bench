def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    import numpy as np
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=float)
    (a, b, c, d, e, f) = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    xi_nodes = [-1, 1, 1, -1, 0, 1, 0, -1]
    eta_nodes = [-1, -1, 1, 1, -1, 0, 1, 0]
    node_values = np.array([a + b * xi + c * eta + d * xi * xi + e * xi * eta + f * eta * eta for (xi, eta) in zip(xi_nodes, eta_nodes)])
    xi_test = np.array([-0.5, 0.0, 0.5])
    eta_test = np.array([0.2, -0.3, 0.4])
    grad_analytic_x = b + 2 * d * xi_test + e * eta_test
    grad_analytic_y = c + e * xi_test + 2 * f * eta_test
    grad_expected = np.vstack([grad_analytic_x, grad_analytic_y])
    grad_computed = fcn(node_coords, node_values, xi_test, eta_test)
    assert np.allclose(grad_computed, grad_expected, atol=1e-10)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    import numpy as np
    node_coords = np.array([[-1.0, -1.0], [1.5, -0.8], [1.2, 1.3], [-0.8, 1.1], [0.2, -1.0], [1.3, 0.2], [-0.1, 1.2], [-1.0, 0.1]], dtype=float)
    (alpha, beta, gamma) = (0.5, 1.2, -0.7)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    xi_test = np.array([-0.9, -0.3, 0.0, 0.4, 0.8])
    eta_test = np.array([0.7, -0.5, 0.2, -0.1, 0.6])
    grad_expected = np.array([[beta], [gamma]]) * np.ones((2, len(xi_test)))
    grad_computed = fcn(node_coords, node_values, xi_test, eta_test)
    assert np.allclose(grad_computed, grad_expected, atol=1e-10)