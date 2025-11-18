def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη², 
    the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (a, b, c, d, e, f) = (2, 3, -1, 1, 2, 3)
    node_values = np.array([a + b * -1 + c * -1 + d * 1 + e * 1 + f * 1, a + b * 1 + c * -1 + d * 1 - e * 1 + f * 1, a + b * 1 + c * 1 + d * 1 + e * 1 + f * 1, a + b * -1 + c * 1 + d * 1 - e * 1 + f * 1, a + b * 0 + c * -1 + d * 0 + e * 0 + f * 1, a + b * 1 + c * 0 + d * 1 + e * 0 + f * 0, a + b * 0 + c * 1 + d * 0 + e * 0 + f * 1, a + b * -1 + c * 0 + d * 1 + e * 0 + f * 0])
    xi = np.array([-0.5, 0.25, 0.8])
    eta = np.array([0.1, -0.7, 0.3])
    grad_num = fcn(node_coords, node_values, xi, eta)
    grad_exact = np.zeros((2, len(xi)))
    for i in range(len(xi)):
        grad_exact[0, i] = b + 2 * d * xi[i] + e * eta[i]
        grad_exact[1, i] = c + e * xi[i] + 2 * f * eta[i]
    assert np.allclose(grad_num, grad_exact)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[-1, -1], [1, -1], [1.2, 1.1], [-0.8, 0.9], [0, -1.2], [1.1, 0], [0.2, 1], [-0.9, 0]])
    (alpha, beta, gamma) = (2, 3, -1)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    xi = np.array([-0.7, 0.0, 0.4])
    eta = np.array([0.2, -0.5, 0.8])
    grad = fcn(node_coords, node_values, xi, eta)
    grad_exact = np.array([[beta, beta, beta], [gamma, gamma, gamma]])
    assert np.allclose(grad, grad_exact)