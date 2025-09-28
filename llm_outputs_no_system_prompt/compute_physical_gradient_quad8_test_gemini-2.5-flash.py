def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """
    Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    xi_nodes = node_coords[:, 0]
    eta_nodes = node_coords[:, 1]
    node_values = 1 + 2 * xi_nodes + 3 * eta_nodes + 4 * xi_nodes ** 2 + 5 * xi_nodes * eta_nodes + 6 * eta_nodes ** 2
    xi_eval = np.array([0.0, 0.5, -0.2, 0.75, -0.8, 0.1])
    eta_eval = np.array([0.0, 0.5, 0.8, -0.25, 0.9, -0.1])
    grad_x_expected = 2 + 8 * xi_eval + 5 * eta_eval
    grad_y_expected = 3 + 5 * xi_eval + 12 * eta_eval
    grad_phys_expected = np.array([grad_x_expected, grad_y_expected])
    grad_phys_actual = fcn(node_coords, node_values, xi_eval, eta_eval)
    npt.assert_allclose(grad_phys_actual, grad_phys_expected, rtol=1e-10, atol=1e-10)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """
    Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -0.8], [0.8, 1.0], [-1.2, 1.0], [0.0, -1.1], [1.1, 0.0], [0.0, 1.2], [-1.1, 0.0]])
    alpha = 5.0
    beta = 2.0
    gamma = -3.0
    x_nodes = node_coords[:, 0]
    y_nodes = node_coords[:, 1]
    node_values = alpha + beta * x_nodes + gamma * y_nodes
    xi_eval = np.array([0.0, 0.5, -0.2, 0.75, -0.8, 0.1, -0.9, 0.9])
    eta_eval = np.array([0.0, 0.5, 0.8, -0.25, 0.9, -0.1, -0.7, 0.7])
    n_pts = len(xi_eval)
    grad_x_expected = np.full(n_pts, beta)
    grad_y_expected = np.full(n_pts, gamma)
    grad_phys_expected = np.array([grad_x_expected, grad_y_expected])
    grad_phys_actual = fcn(node_coords, node_values, xi_eval, eta_eval)
    npt.assert_allclose(grad_phys_actual, grad_phys_expected, rtol=1e-10, atol=1e-10)