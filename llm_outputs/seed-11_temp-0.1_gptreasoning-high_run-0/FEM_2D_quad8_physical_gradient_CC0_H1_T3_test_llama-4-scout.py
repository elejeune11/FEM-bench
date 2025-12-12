def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """
    Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient.
    """
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (a, b, c, d, e, f) = (1, 2, 3, 0.1, 0.2, 0.3)
    u = lambda xi, eta: a + b * xi + c * eta + d * xi ** 2 + e * xi * eta + f * eta ** 2
    node_values = u(node_coords[:, 0], node_coords[:, 1])
    (xi, eta) = (0.5, 0.5)
    grad_phys = fcn(node_coords, node_values, xi, eta)
    du_dx_ana = b + 2 * d * xi + e * eta
    du_dy_ana = c + e * xi + 2 * f * eta
    grad_phys_ana = np.array([du_dx_ana, du_dy_ana])
    assert np.allclose(grad_phys, grad_phys_ana)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """
    Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points.
    """
    node_coords = np.array([[-1, -1], [2, -1], [2, 2], [-1, 2], [0.5, -1], [2, 0.5], [0.5, 2], [-1, 0.5]])
    (alpha, beta, gamma) = (1, 2, 3)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    xi = np.array([0.5, -0.5])
    eta = np.array([0.5, -0.5])
    grad_phys = fcn(node_coords, node_values, xi, eta)
    grad_phys_ana = np.array([[beta], [gamma]])
    assert np.allclose(grad_phys, grad_phys_ana)