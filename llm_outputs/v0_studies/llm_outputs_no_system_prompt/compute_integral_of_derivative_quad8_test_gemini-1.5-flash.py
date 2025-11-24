def test_integral_of_derivative_quad8_identity_cubic(fcn):
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    node_values = np.array([-2, 0, 2, 0, -1, 1, 1, -1]) ** 3 + np.array([-2, -2, 2, 2, -1, 0, 1, 0]) ** 3
    integral = fcn(node_coords, node_values, 4)
    assert np.allclose(integral, [4, 4])

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    A = np.array([[2, 0], [0, 3]])
    c = np.array([1, 2])
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]]) @ A.T + c
    (alpha, beta, gamma) = (1, 2, 3)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    integral = fcn(node_coords, node_values, 1)
    assert np.allclose(integral, [beta * 4 * np.abs(np.linalg.det(A)), gamma * 4 * np.abs(np.linalg.det(A))])

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-0.2, -1], [1, -0.3], [0.1, 1], [-1, 0.4]])
    node_values = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    integral_4 = fcn(node_coords, node_values, 4)
    integral_9 = fcn(node_coords, node_values, 9)
    assert not np.allclose(integral_4, integral_9)