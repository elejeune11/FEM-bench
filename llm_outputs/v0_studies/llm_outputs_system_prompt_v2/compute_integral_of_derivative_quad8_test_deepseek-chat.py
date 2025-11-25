def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """Analytic check with identity mapping (reference element = physical element)."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    node_values = node_coords[:, 0] ** 3 + node_coords[:, 1] ** 3
    result = fcn(node_coords, node_values, 4)
    expected = np.array([4.0, 4.0])
    assert np.allclose(result, expected, rtol=1e-10)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """Analytic check with an affine geometric map and a linear field."""
    A = np.array([[2.0, 0.5], [0.25, 1.5]])
    c = np.array([1.0, -0.5])
    ref_nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    node_coords = (A @ ref_nodes.T).T + c
    (beta, gamma) = (2.0, 3.0)
    node_values = beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    det_A = np.linalg.det(A)
    area = 4.0 * abs(det_A)
    expected = np.array([beta, gamma]) * area
    result = fcn(node_coords, node_values, 4)
    assert np.allclose(result, expected, rtol=1e-10)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """Test quadrature-order sensitivity on a deliberately curved, asymmetric mapping."""
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    mid_nodes = np.array([[0, -1.2], [1.3, 0], [0, 0.8], [-0.7, 0]])
    node_coords = np.vstack([corners, mid_nodes])
    node_values = node_coords[:, 0] ** 3 + 2 * node_coords[:, 0] * node_coords[:, 1] ** 2 + node_coords[:, 1] ** 2
    result_1pt = fcn(node_coords, node_values, 1)
    result_4pt = fcn(node_coords, node_values, 4)
    result_9pt = fcn(node_coords, node_values, 9)
    assert not np.allclose(result_1pt, result_4pt, rtol=1e-10)
    assert not np.allclose(result_4pt, result_9pt, rtol=1e-10)