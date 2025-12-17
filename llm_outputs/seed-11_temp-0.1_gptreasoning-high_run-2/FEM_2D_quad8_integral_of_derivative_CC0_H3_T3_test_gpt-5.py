def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """
    Validate ∫_Ω (∇u) dΩ for a cubic field on an identity-mapped Q8 element.
    A 2×2 Gauss–Legendre rule (num_gauss_pts = 4) is exact for this case.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)

    def u_fun(x, y):
        return x ** 3 + 2 * x ** 2 * y - x * y ** 2 + 3 * y ** 3 + 4 * x - 5 * y + 1
    node_values = np.array([u_fun(x, y) for x, y in node_coords], dtype=float)
    expected = np.array([56.0 / 3.0, -16.0 / 3.0], dtype=float)
    result = fcn(node_coords, node_values, 4)
    assert result.shape == (2,)
    assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """
    Analytic check with an affine geometric map and a linear field.
    Map the reference square Q by [x, y]^T = A[ξ, η]^T + c.
    For u(x, y) = α + βx + γy, ∫_Ω ∇u dΩ = [β, γ] * area, area = 4 * det(A).
    """
    ref_nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    A = np.array([[1.2, 0.4], [-0.3, 1.7]], dtype=float)
    c = np.array([0.7, -1.2], dtype=float)
    detA = np.linalg.det(A)
    area = 4.0 * detA
    alpha = -1.4
    beta = 0.8
    gamma = -0.6
    node_coords = ref_nodes @ A.T + c
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    expected = np.array([beta * area, gamma * area], dtype=float)
    for ng in (1, 4, 9):
        result = fcn(node_coords, node_values, ng)
        assert result.shape == (2,)
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """
    Check quadrature-order sensitivity on a curved, asymmetric Q8 mapping.
    Use a non-affine geometry and asymmetric nodal values so that 1×1 quadrature
    is insufficient, while 2×2 and 3×3 agree. Confirms higher-order integration
    sensitivity to nonlinear mappings.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.2], [1.3, 0.2], [0.2, 1.4], [-1.4, -0.3]], dtype=float)

    def u_poly(x, y):
        return 1.0 + 0.5 * x - 0.3 * y + 0.8 * x ** 2 + 0.3 * x * y + 0.7 * y ** 2
    node_values = np.array([u_poly(x, y) for x, y in node_coords], dtype=float)
    res_1 = fcn(node_coords, node_values, 1)
    res_4 = fcn(node_coords, node_values, 4)
    res_9 = fcn(node_coords, node_values, 9)
    assert not np.allclose(res_9, res_1, rtol=1e-10, atol=1e-10)
    assert np.allclose(res_4, res_9, rtol=1e-12, atol=1e-12)