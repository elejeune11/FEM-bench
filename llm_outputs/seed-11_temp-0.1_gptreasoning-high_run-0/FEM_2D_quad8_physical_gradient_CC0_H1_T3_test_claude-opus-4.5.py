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
    test_points = [(0.0, 0.0), (0.5, 0.5), (-0.5, 0.5), (0.3, -0.7), (-0.8, -0.2)]
    for (xi, eta) in test_points:
        grad_phys = fcn(node_coords, node_values, xi, eta)
        grad_expected = grad_u_analytic(xi, eta)
        assert grad_phys.shape == (2, 1) or grad_phys.shape == (2,), f'Unexpected shape {grad_phys.shape}'
        grad_phys_flat = grad_phys.flatten()[:2]
        assert np.allclose(grad_phys_flat, grad_expected, rtol=1e-10, atol=1e-12), f'At ({xi}, {eta}): expected {grad_expected}, got {grad_phys_flat}'
    xi_arr = np.array([0.0, 0.5, -0.5])
    eta_arr = np.array([0.0, 0.5, 0.5])
    grad_phys_arr = fcn(node_coords, node_values, xi_arr, eta_arr)
    assert grad_phys_arr.shape == (2, 3), f'Unexpected shape {grad_phys_arr.shape}'
    for i in range(3):
        grad_expected = grad_u_analytic(xi_arr[i], eta_arr[i])
        assert np.allclose(grad_phys_arr[:, i], grad_expected, rtol=1e-10, atol=1e-12)

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """
    Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points.
    """
    node_coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.5, 2.0], [0.0, 2.0], [1.0, -0.2], [2.3, 1.0], [1.25, 2.1], [-0.1, 1.0]])
    (alpha, beta, gamma) = (5.0, 3.0, -2.0)

    def u_linear(x, y):
        return alpha + beta * x + gamma * y
    node_values = np.array([u_linear(x, y) for (x, y) in node_coords])
    grad_expected = np.array([beta, gamma])
    test_points = [(0.0, 0.0), (0.5, 0.5), (-0.5, 0.5), (0.3, -0.7), (-0.8, -0.2), (0.9, 0.9), (-0.9, -0.9)]
    for (xi, eta) in test_points:
        grad_phys = fcn(node_coords, node_values, xi, eta)
        grad_phys_flat = grad_phys.flatten()[:2]
        assert np.allclose(grad_phys_flat, grad_expected, rtol=1e-10, atol=1e-10), f'At ({xi}, {eta}): expected {grad_expected}, got {grad_phys_flat}'
    xi_arr = np.array([-0.5, 0.0, 0.5, 0.7])
    eta_arr = np.array([-0.5, 0.0, 0.5, -0.3])
    grad_phys_arr = fcn(node_coords, node_values, xi_arr, eta_arr)
    assert grad_phys_arr.shape == (2, 4), f'Unexpected shape {grad_phys_arr.shape}'
    for i in range(4):
        assert np.allclose(grad_phys_arr[:, i], grad_expected, rtol=1e-10, atol=1e-10), f'At point {i}: expected {grad_expected}, got {grad_phys_arr[:, i]}'