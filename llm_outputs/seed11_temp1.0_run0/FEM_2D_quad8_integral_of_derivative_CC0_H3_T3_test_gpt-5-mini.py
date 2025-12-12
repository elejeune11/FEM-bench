def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """Validate ∫_Ω (∇u) dΩ for a cubic field on an identity-mapped Q8 element.
    A 2×2 Gauss–Legendre rule (num_gauss_pts = 4) is exact for this case."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)

    def u_scalar(x, y):
        return x ** 3 + x ** 2 * y + x * y ** 2 + y ** 3
    node_values = np.array([u_scalar(x, y) for (x, y) in node_coords], dtype=float)
    result = fcn(node_coords, node_values, 4)
    expected = np.array([16.0 / 3.0, 20.0 / 3.0], dtype=float)
    assert np.allclose(result, expected, rtol=1e-08, atol=1e-10)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """Analytic check with an affine geometric map and a linear field.
    Map the reference square Q by x = A [ξ,η] + c. For u(x,y) = α + β x + γ y,
    verify the returned integral equals [β * area(Ω), γ * area(Ω)]."""
    ref_nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    A = np.array([[2.0, 0.5], [0.3, 1.5]], dtype=float)
    c = np.array([0.7, -0.4], dtype=float)
    node_coords = ref_nodes @ A.T + c
    alpha = 1.23
    beta = -0.75
    gamma = 2.5
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    detA = np.linalg.det(A)
    area = abs(detA) * 4.0
    expected = np.array([beta * area, gamma * area], dtype=float)
    result = fcn(node_coords, node_values, 1)
    assert np.allclose(result, expected, rtol=1e-08, atol=1e-10)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """Check quadrature-order sensitivity on a curved, asymmetric Q8 mapping.
    For a non-affine geometry and a higher-order scalar field, the integral
    computed with a 3×3 rule should differ from 1×1 or 2×2 rules."""
    node_coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 2.0], [1.0, -0.5], [2.2, 1.4], [1.0, 2.7], [-0.2, 1.1]], dtype=float)

    def u_scalar(x, y):
        return x ** 4 + y ** 4 + 2.0 * x ** 3 * y - 1.3 * x * y ** 3 + 0.5 * x ** 2 * y ** 2
    node_values = np.array([u_scalar(x, y) for (x, y) in node_coords], dtype=float)
    res_1 = fcn(node_coords, node_values, 1)
    res_4 = fcn(node_coords, node_values, 4)
    res_9 = fcn(node_coords, node_values, 9)
    assert not np.allclose(res_9, res_4, rtol=1e-08, atol=1e-10)
    assert not np.allclose(res_9, res_1, rtol=1e-08, atol=1e-10)