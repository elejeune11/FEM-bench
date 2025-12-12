def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient.
    """
    import numpy as np
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (a, b, c, d, e, f) = (0.3, -0.5, 0.8, 0.2, -0.3, 0.1)
    xi_nodes = node_coords[:, 0]
    eta_nodes = node_coords[:, 1]
    node_values = a + b * xi_nodes + c * eta_nodes + d * xi_nodes ** 2 + e * xi_nodes * eta_nodes + f * eta_nodes ** 2
    xis = np.array([-0.9, -0.1, 0.2, 0.7])
    etas = np.array([-0.8, 0.0, 0.4, 0.6])
    grad = fcn(node_coords, node_values, xis, etas)
    du_dxi = b + 2.0 * d * xis + e * etas
    du_deta = c + e * xis + 2.0 * f * etas
    expected = np.vstack([du_dxi, du_deta])
    assert grad.shape == expected.shape
    assert np.allclose(grad, expected, rtol=1e-08, atol=1e-08)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points.
    """
    import numpy as np
    corners = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    mid5 = (corners[0] + corners[1]) / 2.0 + np.array([0.1, -0.05])
    mid6 = (corners[1] + corners[2]) / 2.0 + np.array([0.05, 0.08])
    mid7 = (corners[2] + corners[3]) / 2.0 + np.array([-0.07, 0.04])
    mid8 = (corners[3] + corners[0]) / 2.0 + np.array([-0.02, -0.03])
    node_coords = np.vstack([corners[0], corners[1], corners[2], corners[3], mid5, mid6, mid7, mid8])
    alpha = 1.2
    beta = -0.7
    gamma = 2.3
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    xis = np.array([-0.8, -0.3, 0.0, 0.5, 0.9])
    etas = np.array([-0.9, -0.25, 0.2, 0.6, 0.95])
    grad = fcn(node_coords, node_values, xis, etas)
    expected = np.vstack([np.full_like(xis, beta, dtype=float), np.full_like(etas, gamma, dtype=float)])
    assert grad.shape == expected.shape
    assert np.allclose(grad, expected, rtol=1e-08, atol=1e-08)