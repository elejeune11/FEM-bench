def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient."""
    import numpy as np
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (a, b, c, d, e, f) = (2, 3, 4, 5, 6, 7)
    xi_nodes = node_coords[:, 0]
    eta_nodes = node_coords[:, 1]
    node_values = a + b * xi_nodes + c * eta_nodes + d * xi_nodes ** 2 + e * xi_nodes * eta_nodes + f * eta_nodes ** 2
    xi_test = np.array([0.0, -0.5, 0.5, 0.25, -0.75])
    eta_test = np.array([0.0, 0.5, -0.5, 0.75, -0.25])
    grad_phys = fcn(node_coords, node_values, xi_test, eta_test)
    expected_dudx = b + 2 * d * xi_test + e * eta_test
    expected_dudy = c + e * xi_test + 2 * f * eta_test
    assert np.allclose(grad_phys[0, :], expected_dudx, rtol=1e-10, atol=1e-12)
    assert np.allclose(grad_phys[1, :], expected_dudy, rtol=1e-10, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    import numpy as np
    node_coords = np.array([[0.0, 0.0], [2.0, 0.1], [2.2, 2.0], [0.1, 1.9], [1.0, 0.0], [2.1, 1.05], [1.15, 2.05], [0.0, 0.95]])
    (alpha, beta, gamma) = (3.0, 5.0, 7.0)
    x_nodes = node_coords[:, 0]
    y_nodes = node_coords[:, 1]
    node_values = alpha + beta * x_nodes + gamma * y_nodes
    xi_test = np.array([0.0, -0.3, 0.6, -0.8, 0.9, -0.5, 0.2])
    eta_test = np.array([0.0, 0.4, -0.7, 0.8, -0.9, 0.5, -0.2])
    grad_phys = fcn(node_coords, node_values, xi_test, eta_test)
    expected_grad = np.array([[beta], [gamma]])
    expected_grad = np.tile(expected_grad, (1, len(xi_test)))
    assert np.allclose(grad_phys, expected_grad, rtol=1e-10, atol=1e-12)