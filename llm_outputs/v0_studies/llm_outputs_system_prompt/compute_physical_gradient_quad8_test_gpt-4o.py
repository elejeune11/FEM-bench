def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """
    Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient.
    """
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (a, b, c, d, e, f) = (1, 2, 3, 4, 5, 6)
    node_values = np.array([a + b * -1 + c * -1 + d * (-1) ** 2 + e * -1 * -1 + f * (-1) ** 2, a + b * 1 + c * -1 + d * 1 ** 2 + e * 1 * -1 + f * (-1) ** 2, a + b * 1 + c * 1 + d * 1 ** 2 + e * 1 * 1 + f * 1 ** 2, a + b * -1 + c * 1 + d * (-1) ** 2 + e * -1 * 1 + f * 1 ** 2, a + b * 0 + c * -1 + d * 0 ** 2 + e * 0 * -1 + f * (-1) ** 2, a + b * 1 + c * 0 + d * 1 ** 2 + e * 1 * 0 + f * 0 ** 2, a + b * 0 + c * 1 + d * 0 ** 2 + e * 0 * 1 + f * 1 ** 2, a + b * -1 + c * 0 + d * (-1) ** 2 + e * -1 * 0 + f * 0 ** 2])
    test_points = np.array([[0, 0], [0.5, -0.5], [-0.5, 0.5]])
    expected_gradients = np.array([[b + 2 * d * xi + e * eta, c + e * xi + 2 * f * eta] for (xi, eta) in test_points]).T
    computed_gradients = fcn(node_coords, node_values, test_points[:, 0], test_points[:, 1])
    assert np.allclose(computed_gradients, expected_gradients), 'Computed gradients do not match expected analytic gradients.'

def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """
    Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points.
    """
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (α, β, γ) = (1, 2, 3)
    node_values = α + β * node_coords[:, 0] + γ * node_coords[:, 1]
    test_points = np.array([[0, 0], [0.5, -0.5], [-0.5, 0.5]])
    expected_gradient = np.array([β, γ]).reshape(2, 1)
    computed_gradients = fcn(node_coords, node_values, test_points[:, 0], test_points[:, 1])
    for i in range(computed_gradients.shape[1]):
        assert np.allclose(computed_gradients[:, i], expected_gradient[:, 0]), f'Computed gradient at point {i} does not match expected gradient.'