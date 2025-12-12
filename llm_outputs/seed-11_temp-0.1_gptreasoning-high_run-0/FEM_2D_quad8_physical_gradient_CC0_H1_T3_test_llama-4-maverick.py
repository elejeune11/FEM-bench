def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """
    Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient.
    """
    (a, b, c, d, e, f) = (1, 2, 3, 4, 5, 6)
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=float)
    xi = np.linspace(-1, 1, 5)
    eta = np.linspace(-1, 1, 5)
    (xi_grid, eta_grid) = np.meshgrid(xi, eta)
    xi_grid = xi_grid.flatten()
    eta_grid = eta_grid.flatten()
    u_analytic = a + b * xi_grid + c * eta_grid + d * xi_grid ** 2 + e * xi_grid * eta_grid + f * eta_grid ** 2
    du_dxi_analytic = b + 2 * d * xi_grid + e * eta_grid
    du_deta_analytic = c + e * xi_grid + 2 * f * eta_grid
    node_values = a + b * node_coords[:, 0] + c * node_coords[:, 1] + d * node_coords[:, 0] ** 2 + e * node_coords[:, 0] * node_coords[:, 1] + f * node_coords[:, 1] ** 2
    grad_phys = fcn(node_coords, node_values, xi_grid, eta_grid)
    grad_analytic = np.vstack((du_dxi_analytic, du_deta_analytic))
    assert np.allclose(grad_phys, grad_analytic, atol=1e-10)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """
    Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points.
    """
    (alpha, beta, gamma) = (1, 2, 3)
    node_coords = np.array([[-1.1, -1.2], [1.3, -1.1], [1.2, 1.1], [-1.3, 1.2], [0, -1.1], [1.2, 0], [0, 1.1], [-1.2, 0]], dtype=float)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    xi = np.linspace(-1, 1, 5)
    eta = np.linspace(-1, 1, 5)
    (xi_grid, eta_grid) = np.meshgrid(xi, eta)
    xi_grid = xi_grid.flatten()
    eta_grid = eta_grid.flatten()
    grad_phys = fcn(node_coords, node_values, xi_grid, eta_grid)
    grad_expected = np.tile(np.array([[beta], [gamma]]), (1, len(xi_grid)))
    assert np.allclose(grad_phys, grad_expected, atol=1e-10)