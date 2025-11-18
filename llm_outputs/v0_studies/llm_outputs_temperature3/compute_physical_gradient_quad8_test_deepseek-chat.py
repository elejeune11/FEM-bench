def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    node_values = np.array([2 + 3 * -1 - 4 * -1 + (-1) ** 2 + 2 * -1 * -1 - 3 * (-1) ** 2, 2 + 3 * 1 - 4 * -1 + 1 ** 2 + 2 * 1 * -1 - 3 * (-1) ** 2, 2 + 3 * 1 - 4 * 1 + 1 ** 2 + 2 * 1 * 1 - 3 * 1 ** 2, 2 + 3 * -1 - 4 * 1 + (-1) ** 2 + 2 * -1 * 1 - 3 * 1 ** 2, 2 + 3 * 0 - 4 * -1 + 0 ** 2 + 2 * 0 * -1 - 3 * (-1) ** 2, 2 + 3 * 1 - 4 * 0 + 1 ** 2 + 2 * 1 * 0 - 3 * 0 ** 2, 2 + 3 * 0 - 4 * 1 + 0 ** 2 + 2 * 0 * 1 - 3 * 1 ** 2, 2 + 3 * -1 - 4 * 0 + (-1) ** 2 + 2 * -1 * 0 - 3 * 0 ** 2])
    test_points = [(0, 0), (0.5, -0.5), (-0.3, 0.7)]
    for (xi, eta) in test_points:
        grad_computed = fcn(node_coords, node_values, xi, eta)
        grad_analytic = np.array([3 + 2 * xi + 2 * eta, -4 + 2 * xi - 6 * eta])
        np.testing.assert_allclose(grad_computed.flatten(), grad_analytic, rtol=1e-10)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points."""
    node_coords = np.array([[0, 0], [3, 0.5], [2.5, 3], [0.5, 2.5], [1.5, 0.2], [2.8, 1.5], [1.2, 2.8], [0.2, 1.2]])
    (alpha, beta, gamma) = (2, 3, -4)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    test_points = [(0, 0), (0.5, -0.5), (-0.7, 0.3), (0.2, -0.8)]
    for (xi, eta) in test_points:
        grad_computed = fcn(node_coords, node_values, xi, eta)
        expected_gradient = np.array([beta, gamma])
        np.testing.assert_allclose(grad_computed.flatten(), expected_gradient, rtol=1e-10)