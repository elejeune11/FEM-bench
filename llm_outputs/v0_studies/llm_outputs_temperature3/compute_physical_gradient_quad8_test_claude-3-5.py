def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη², 
    the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (a, b, c, d, e, f) = (1, 2, 3, 4, 5, 6)
    node_values = np.array([a + b * -1 + c * -1 + d * 1 + e * 1 + f * 1, a + b * 1 + c * -1 + d * 1 - e * 1 + f * 1, a + b * 1 + c * 1 + d * 1 + e * 1 + f * 1, a + b * -1 + c * 1 + d * 1 - e * 1 + f * 1, a + c * -1 + f * 1, a + b * 1 + d * 1, a + c * 1 + f * 1, a + b * -1 + d * 1])
    xi = np.array([-0.5, 0.0, 0.7])
    eta = np.array([0.3, -0.2, 0.1])
    grad = fcn(node_coords, node_values, xi, eta)
    grad_analytic = np.array([[2 + 8 * xi + 5 * eta], [3 + 5 * xi + 12 * eta]]).reshape(2, -1)
    assert np.allclose(grad, grad_analytic)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[0, 0], [2, 0], [2.5, 2], [0.5, 1.8], [1, 0], [2.2, 1], [1.5, 2], [0.2, 1]])
    (alpha, beta, gamma) = (1, 2, 3)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    xi = np.array([-0.7, 0.1, 0.8])
    eta = np.array([0.2, -0.4, 0.5])
    grad = fcn(node_coords, node_values, xi, eta)
    grad_exact = np.array([[beta], [gamma]]).repeat(len(xi), axis=1)
    assert np.allclose(grad, grad_exact)