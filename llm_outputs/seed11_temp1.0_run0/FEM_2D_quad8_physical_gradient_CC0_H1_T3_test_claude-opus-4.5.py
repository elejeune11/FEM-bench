def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """
    Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (a, b, c, d, e, f) = (1.0, 2.0, 3.0, 0.5, 1.5, 0.7)

    def u_func(x, y):
        return a + b * x + c * y + d * x ** 2 + e * x * y + f * y ** 2
    node_values = np.array([u_func(x, y) for (x, y) in node_coords])

    def grad_u_analytic(x, y):
        return np.array([b + 2 * d * x + e * y, c + e * x + 2 * f * y])
    test_points_xi = np.array([0.0, 0.5, -0.5, 0.25, -0.75])
    test_points_eta = np.array([0.0, 0.5, -0.5, -0.25, 0.75])
    grad_phys = fcn(node_coords, node_values, test_points_xi, test_points_eta)
    for j in range(len(test_points_xi)):
        xi_j = test_points_xi[j]
        eta_j = test_points_eta[j]
        expected_grad = grad_u_analytic(xi_j, eta_j)
        computed_grad = grad_phys[:, j]
        assert np.allclose(computed_grad, expected_grad, rtol=1e-10, atol=1e-10), f'Gradient mismatch at (ξ={xi_j}, η={eta_j}): computed={computed_grad}, expected={expected_grad}'

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """
    Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points.
    """
    node_coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.5, 2.0], [0.0, 2.0], [1.0, -0.2], [2.4, 1.0], [1.2, 2.3], [-0.1, 1.0]])
    (alpha, beta, gamma) = (5.0, 3.0, -2.0)

    def u_linear(x, y):
        return alpha + beta * x + gamma * y
    node_values = np.array([u_linear(x, y) for (x, y) in node_coords])
    expected_grad = np.array([beta, gamma])
    test_points_xi = np.array([0.0, 0.5, -0.5, 0.25, -0.75, 0.8, -0.8])
    test_points_eta = np.array([0.0, 0.5, -0.5, -0.25, 0.75, -0.3, 0.6])
    grad_phys = fcn(node_coords, node_values, test_points_xi, test_points_eta)
    for j in range(len(test_points_xi)):
        computed_grad = grad_phys[:, j]
        assert np.allclose(computed_grad, expected_grad, rtol=1e-10, atol=1e-10), f'Gradient mismatch at point {j}: computed={computed_grad}, expected={expected_grad}'