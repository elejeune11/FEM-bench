def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Identity isoparametric mapping: for u(ξ,η)=a+bξ+cη+dξ²+eξη+fη², the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (a, b, c, d, e, f) = (0.7, -0.3, 1.1, 0.5, -0.2, 0.8)
    xi_nodes = node_coords[:, 0]
    eta_nodes = node_coords[:, 1]
    node_values = a + b * xi_nodes + c * eta_nodes + d * xi_nodes ** 2 + e * xi_nodes * eta_nodes + f * eta_nodes ** 2
    xi_eval = np.array([-0.7, -0.2, 0.0, 0.3, 0.6, 0.9, -0.4])
    eta_eval = np.array([-0.5, 0.4, 0.0, -0.8, 0.7, 0.1, 0.2])
    grad = fcn(node_coords, node_values, xi_eval, eta_eval)
    grad = np.asarray(grad)
    if grad.ndim == 1:
        grad = grad.reshape(2, -1)
    if grad.shape[0] != 2 and grad.shape[1] == 2:
        grad = grad.T
    grad_exact = np.vstack([b + 2.0 * d * xi_eval + e * eta_eval, c + e * xi_eval + 2.0 * f * eta_eval])
    assert grad.shape == grad_exact.shape
    assert np.allclose(grad, grad_exact, rtol=1e-12, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Under any isoparametric mapping, a linear physical field u(x,y)=α+βx+γy has constant physical gradient [β, γ]^T."""
    node_coords = np.array([[-1.0, -1.0], [2.0, -0.5], [1.8, 2.2], [-1.2, 1.8], [0.2, -1.2], [2.2, 0.3], [0.1, 2.4], [-1.4, 0.1]])
    (alpha, beta, gamma) = (2.3, -1.7, 0.9)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    xi_grid = np.array([-0.75, 0.0, 0.6])
    eta_grid = np.array([-0.6, 0.2, 0.7])
    (XI, ETA) = np.meshgrid(xi_grid, eta_grid, indexing='xy')
    xi_eval = XI.ravel()
    eta_eval = ETA.ravel()
    grad = fcn(node_coords, node_values, xi_eval, eta_eval)
    grad = np.asarray(grad)
    if grad.ndim == 1:
        grad = grad.reshape(2, -1)
    if grad.shape[0] != 2 and grad.shape[1] == 2:
        grad = grad.T
    grad_exact = np.vstack([np.full_like(xi_eval, beta, dtype=float), np.full_like(xi_eval, gamma, dtype=float)])
    assert grad.shape == grad_exact.shape
    assert np.allclose(grad, grad_exact, rtol=1e-12, atol=1e-12)