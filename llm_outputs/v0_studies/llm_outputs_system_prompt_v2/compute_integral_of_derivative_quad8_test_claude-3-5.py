def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """Analytic check with identity mapping (reference element = physical element)."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    node_values = np.array([(-1) ** 3 + (-1) ** 3, 1 - 1, 1 + 1, -1 + 1, 0 - 1, 1 + 0, 0 + 1, -1 + 0])
    result = fcn(node_coords, node_values, 4)
    expected = np.array([4.0, 4.0])
    assert np.allclose(result, expected, rtol=1e-14, atol=1e-14)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """Analytic check with an affine geometric map and a linear field."""
    A = np.array([[2.0, 0.5], [-0.5, 1.5]])
    c = np.array([1.0, -1.0])
    ref_nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    node_coords = ref_nodes @ A.T + c
    (beta, gamma) = (2.0, -3.0)
    node_values = beta * node_coords[:, 0] + gamma * node_coords[:, 1] + 1.0
    result = fcn(node_coords, node_values, 4)
    det_A = np.linalg.det(A)
    expected = np.array([beta, gamma]) * (4.0 * abs(det_A))
    assert np.allclose(result, expected, rtol=1e-14, atol=1e-14)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """Test quadrature-order sensitivity on a deliberately curved, asymmetric mapping."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0.2, -1.1], [1.1, 0.3], [0.1, 1.2], [-0.9, 0.1]])
    node_values = np.array([1.0, -0.5, 2.0, 0.8, 1.5, -0.3, 0.7, 1.2])
    result_1pt = fcn(node_coords, node_values, 1)
    result_4pt = fcn(node_coords, node_values, 4)
    result_9pt = fcn(node_coords, node_values, 9)
    assert not np.allclose(result_4pt, result_1pt, rtol=1e-10, atol=1e-10)
    assert not np.allclose(result_9pt, result_4pt, rtol=1e-10, atol=1e-10)