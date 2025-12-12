def test_integral_of_derivative_quad8_identity_cubic(fcn):
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    node_values = np.array([(-1) ** 3 + (-1) ** 3, 1 ** 3 + (-1) ** 3, 1 ** 3 + 1 ** 3, (-1) ** 3 + 1 ** 3, 0 ** 3 + (-1) ** 3, 1 ** 3 + 0 ** 3, 0 ** 3 + 1 ** 3, (-1) ** 3 + 0 ** 3])
    result = fcn(node_coords, node_values, 4)
    assert np.allclose(result, [4.0, 4.0], atol=1e-10)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    A = np.array([[2.0, 1.0], [0.5, 3.0]])
    c = np.array([1.0, 2.0])
    (α, β, γ) = (5.0, 3.0, -2.0)
    ref_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    node_coords = ref_coords @ A.T + c
    node_values = α + β * node_coords[:, 0] + γ * node_coords[:, 1]
    det_J = np.linalg.det(A)
    area_ref = 4.0
    expected = np.array([β, γ]) * det_J * area_ref
    result = fcn(node_coords, node_values, 4)
    assert np.allclose(result, expected, atol=1e-10)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    node_coords = np.array([[-1.0, -1.0], [1.2, -0.8], [1.1, 1.2], [-0.9, 1.1], [0.1, -1.0], [1.0, 0.2], [-0.1, 1.0], [-1.0, 0.1]])
    node_values = np.array([(-1) ** 2 + (-1) ** 2, 1.2 ** 2 + (-0.8) ** 2, 1.1 ** 2 + 1.2 ** 2, (-0.9) ** 2 + 1.1 ** 2, 0.1 ** 2 + (-1.0) ** 2, 1.0 ** 2 + 0.2 ** 2, (-0.1) ** 2 + 1.0 ** 2, (-1.0) ** 2 + 0.1 ** 2])
    result_1 = fcn(node_coords, node_values, 1)
    result_2 = fcn(node_coords, node_values, 4)
    result_3 = fcn(node_coords, node_values, 9)
    assert not np.allclose(result_1, result_2, atol=1e-08)
    assert not np.allclose(result_2, result_3, atol=1e-08)
    assert not np.allclose(result_1, result_3, atol=1e-08)