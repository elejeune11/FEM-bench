def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """
    Test the identity isoparametric mapping (x=ξ, y=η). For any quadratic
    u(ξ,η)=a+bξ+cη+dξ²+eξη+fη², the physical gradient equals the analytic gradient.
    """
    import numpy as np
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (a, b, c, d, e, f) = (0.5, -1.1, 2.3, 0.7, -0.6, 1.2)
    xi_nodes = node_coords[:, 0]
    eta_nodes = node_coords[:, 1]
    node_values = a + b * xi_nodes + c * eta_nodes + d * xi_nodes ** 2 + e * xi_nodes * eta_nodes + f * eta_nodes ** 2
    xi = np.array([-0.8, -0.3, 0.0, 0.5, 0.9])
    eta = np.array([-0.9, 0.2, -0.2, 0.7, 0.1])
    du_dxi = b + 2.0 * d * xi + e * eta
    du_deta = c + e * xi + 2.0 * f * eta
    expected = np.vstack([du_dxi, du_deta])
    grad_phys = fcn(node_coords, node_values, xi, eta)
    assert grad_phys.shape == expected.shape
    assert np.allclose(grad_phys, expected, rtol=1e-12, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """
    Test that under any isoparametric mapping, a linear physical field u(x,y)=α+βx+γy
    is reproduced exactly by quadratic quadrilateral elements. The physical gradient
    should be [β, γ]^T at all points, even with a curved mapping.
    """
    import numpy as np
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.2], [1.2, 0.0], [0.0, 1.2], [-1.2, 0.0]])
    (alpha, beta, gamma) = (2.0, -0.7, 1.3)
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    node_values = alpha + beta * x + gamma * y
    xi = np.array([-0.6, -0.2, 0.0, 0.3, 0.7, 0.1])
    eta = np.array([0.1, -0.4, 0.2, 0.6, -0.3, 0.8])
    grad_phys = fcn(node_coords, node_values, xi, eta)
    expected = np.vstack([np.full_like(xi, beta, dtype=float), np.full_like(eta, gamma, dtype=float)])
    assert grad_phys.shape == expected.shape
    assert np.allclose(grad_phys, expected, rtol=1e-12, atol=1e-12)