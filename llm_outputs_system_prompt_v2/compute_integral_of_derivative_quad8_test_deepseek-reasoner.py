def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """Analytic check with identity mapping (reference element = physical element)."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    node_values = node_coords[:, 0] ** 3 + node_coords[:, 1] ** 3
    result = fcn(node_coords, node_values, 4)
    expected = np.array([4.0, 4.0])
    assert np.allclose(result, expected, rtol=1e-10)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """Analytic check with affine geometric map and linear field."""
    A = np.array([[2.0, 0.5], [0.0, 1.5]])
    c = np.array([1.0, -2.0])
    corners_ref = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    corners_phys = (A @ corners_ref.T).T + c
    midpoints_ref = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
    midpoints_phys = (A @ midpoints_ref.T).T + c
    node_coords = np.vstack([corners_phys, midpoints_phys])
    (alpha, beta, gamma) = (3.0, 2.0, -1.0)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    detA = np.linalg.det(A)
    area = 4.0 * abs(detA)
    expected = np.array([beta, gamma]) * area
    result = fcn(node_coords, node_values, 4)
    assert np.allclose(result, expected, rtol=1e-10)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """Test quadrature-order sensitivity on curved asymmetric mapping."""
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    midpoints = np.array([[0.2, -1.3], [1.4, 0.1], [-0.3, 1.2], [-1.1, -0.4]])
    node_coords = np.vstack([corners, midpoints])
    node_values = np.array([0.5, 1.2, -0.8, 2.1, 0.7, -1.3, 1.8, -0.2])
    result1 = fcn(node_coords, node_values, 1)
    result4 = fcn(node_coords, node_values, 4)
    result9 = fcn(node_coords, node_values, 9)
    assert not np.allclose(result1, result4, rtol=1e-10)
    assert not np.allclose(result4, result9, rtol=1e-10)