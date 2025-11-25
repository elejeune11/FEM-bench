def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """
    Test the identity isoparametric mapping (x=ξ, y=η).
    For any quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη², the physical gradient equals the analytic gradient.
    """
    rng = np.random.default_rng(123)
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    (a, b, c, d, e, f) = rng.normal(size=6)

    def u_fun(xi, eta):
        return a + b * xi + c * eta + d * xi * xi + e * xi * eta + f * eta * eta
    xis_nodes = node_coords[:, 0]
    etas_nodes = node_coords[:, 1]
    node_values = u_fun(xis_nodes, etas_nodes)
    n_pts = 7
    xi = rng.uniform(-1.0, 1.0, size=n_pts)
    eta = rng.uniform(-1.0, 1.0, size=n_pts)
    grad = fcn(node_coords, node_values, xi, eta)
    assert grad.shape == (2, n_pts)
    grad_x = b + 2.0 * d * xi + e * eta
    grad_y = c + e * xi + 2.0 * f * eta
    expected = np.vstack([grad_x, grad_y])
    assert np.allclose(grad, expected, rtol=1e-12, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """
    Test that under any isoparametric mapping, a linear physical field u(x,y)=α+βx+γy
    is reproduced exactly by quadratic quadrilateral elements, yielding constant physical gradient [β, γ]^T.
    """
    rng = np.random.default_rng(456)
    node_coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, -0.2], [2.1, 1.0], [1.0, 2.2], [-0.1, 1.0]], dtype=float)
    (alpha, beta, gamma) = rng.normal(size=3)

    def u_phys(x, y):
        return alpha + beta * x + gamma * y
    node_values = u_phys(node_coords[:, 0], node_coords[:, 1])
    n_pts = 9
    xi = rng.uniform(-1.0, 1.0, size=n_pts)
    eta = rng.uniform(-1.0, 1.0, size=n_pts)
    grad = fcn(node_coords, node_values, xi, eta)
    assert grad.shape == (2, n_pts)
    expected = np.vstack([np.full(n_pts, beta), np.full(n_pts, gamma)])
    assert np.allclose(grad, expected, rtol=1e-12, atol=1e-12)