def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """Analytic check with identity mapping (reference element = physical element)."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    node_values = node_coords[:, 0] ** 3 + node_coords[:, 1] ** 3
    result = fcn(node_coords, node_values, 4)
    expected = np.array([4.0, 4.0])
    assert np.allclose(result, expected, rtol=1e-10)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """Analytic check with affine geometric map and linear field."""
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    x_corners = 2 * corners[:, 0] + 1
    y_corners = 3 * corners[:, 1] + 2
    corners_phys = np.column_stack([x_corners, y_corners])
    mid5 = 0.5 * (corners_phys[0] + corners_phys[1])
    mid6 = 0.5 * (corners_phys[1] + corners_phys[2])
    mid7 = 0.5 * (corners_phys[2] + corners_phys[3])
    mid8 = 0.5 * (corners_phys[3] + corners_phys[0])
    node_coords = np.vstack([corners_phys, mid5, mid6, mid7, mid8])
    node_values = 2 + 3 * node_coords[:, 0] - 4 * node_coords[:, 1]
    result = fcn(node_coords, node_values, 4)
    expected = np.array([72.0, -96.0])
    assert np.allclose(result, expected, rtol=1e-10)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """Test quadrature-order sensitivity on curved asymmetric mapping."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0.2, -0.8], [0.9, 0.3], [-0.3, 0.7], [-0.8, -0.2]])
    node_values = np.exp(0.1 * node_coords[:, 0]) + 0.5 * node_coords[:, 1] ** 2
    result_1pt = fcn(node_coords, node_values, 1)
    result_4pt = fcn(node_coords, node_values, 4)
    result_9pt = fcn(node_coords, node_values, 9)
    assert not np.allclose(result_9pt, result_4pt, rtol=1e-10)
    assert not np.allclose(result_9pt, result_1pt, rtol=1e-10)