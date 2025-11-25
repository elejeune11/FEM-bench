def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (a, b, c, d, e, f) = (2.0, 3.0, -4.0, 2.0, 5.0, -3.0)
    node_values = np.zeros(8)
    for (i, (xi, eta)) in enumerate(node_coords):
        node_values[i] = a + b * xi + c * eta + d * xi ** 2 + e * xi * eta + f * eta ** 2
    xi = np.array([-0.5, 0.0, 0.7])
    eta = np.array([0.3, -0.2, 0.4])
    grad_computed = fcn(node_coords, node_values, xi, eta)
    grad_exact = np.zeros((2, len(xi)))
    for i in range(len(xi)):
        grad_exact[0, i] = b + 2 * d * xi[i] + e * eta[i]
        grad_exact[1, i] = c + e * xi[i] + 2 * f * eta[i]
    assert_allclose(grad_computed, grad_exact, rtol=1e-14, atol=1e-14)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1.2], [1.1, 0], [0, 1.3], [-1.2, 0]])
    (alpha, beta, gamma) = (3.0, 2.0, -4.0)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    xi = np.array([-0.7, 0.1, 0.8])
    eta = np.array([0.2, -0.3, 0.5])
    grad_computed = fcn(node_coords, node_values, xi, eta)
    grad_exact = np.tile([[beta], [gamma]], (1, len(xi)))
    assert_allclose(grad_computed, grad_exact, rtol=1e-13, atol=1e-13)