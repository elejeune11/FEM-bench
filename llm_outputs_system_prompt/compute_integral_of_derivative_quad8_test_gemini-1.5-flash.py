def test_integral_of_derivative_quad8_identity_cubic(fcn):
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    node_values = np.array([-2, 0, 2, 0, -1, 1, 1, -1])
    num_gauss_pts = 4
    integral = fcn(node_coords, node_values, num_gauss_pts)
    assert np.allclose(integral, [4, 4])

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    node_coords = np.array([[-2, -1], [2, -1], [2, 1], [-2, 1], [0, -1], [2, 0], [0, 1], [-2, 0]])
    node_values = np.array([0, 2, 4, 2, 1, 3, 3, 1])
    num_gauss_pts = 1
    integral = fcn(node_coords, node_values, num_gauss_pts)
    assert np.allclose(integral, [8, 8])

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1.2], [1.1, 0], [0, 1.1], [-1.1, 0]])
    node_values = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    integral_4 = fcn(node_coords, node_values, 4)
    integral_9 = fcn(node_coords, node_values, 9)
    assert not np.allclose(integral_4, integral_9)