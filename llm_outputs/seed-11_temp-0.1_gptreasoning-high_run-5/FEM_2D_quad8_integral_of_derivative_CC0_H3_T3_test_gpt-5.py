def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """
    Validate ∫_Ω (∇u) dΩ for a cubic field on an identity-mapped Q8 element.
    Use u(x,y) = a x^2 y + b x y^2 + c x^2 + d y^2 + e x y + f x + g y + h.
    Since this polynomial lies in the Q8 (serendipity) space, the interpolation is exact.
    A 2×2 Gauss–Legendre rule (num_gauss_pts = 4) integrates the quadratic gradient exactly.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    a, b = (2.0, -1.5)
    c, d, e = (0.3, -0.7, 0.25)
    f, g, h = (1.1, -0.9, 0.2)

    def u(x, y):
        return a * x ** 2 * y + b * x * y ** 2 + c * x ** 2 + d * y ** 2 + e * x * y + f * x + g * y + h
    node_values = np.array([u(x, y) for x, y in node_coords], dtype=float)
    result = fcn(node_coords, node_values, num_gauss_pts=4)
    assert result.shape == (2,)
    expected = np.array([4.0 * b / 3.0 + 4.0 * f, 4.0 * a / 3.0 + 4.0 * g], dtype=float)
    assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """
    Analytic check with an affine geometric map and a linear field.
    Map Q by [x, y]^T = A [ξ, η]^T + c, with det(A) > 0.
    For u(x, y) = α + β x + γ y, ∫_Ω ∇u dΩ = Area * [β, γ],
    where Area = 4 * det(A). The Q8 isoparametric mapping reproduces the affine map exactly.
    """
    A = np.array([[2.0, 0.6], [-0.4, 1.7]], dtype=float)
    c = np.array([0.7, -1.2], dtype=float)
    detA = np.linalg.det(A)
    assert detA > 0.0
    ref_nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    node_coords = ref_nodes @ A.T + c
    alpha, beta, gamma = (1.6, -0.8, 2.3)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    result = fcn(node_coords, node_values, num_gauss_pts=1)
    assert result.shape == (2,)
    area = 4.0 * detA
    expected = np.array([beta * area, gamma * area], dtype=float)
    assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """
    Check quadrature-order sensitivity on a curved, asymmetric Q8 mapping.
    Use a non-affine geometry with asymmetric nodal values. The integral ∫_Ω (∇u) dΩ
    should vary with quadrature order. Confirm that results from a 3×3 rule (9 points)
    differ from those of 1×1 and 2×2 rules.
    """
    node_coords = np.array([[-1.0, -1.0], [1.1, -1.0], [1.0, 1.2], [-1.2, 1.1], [0.1, -1.2], [1.3, 0.2], [-0.2, 1.3], [-1.4, -0.3]], dtype=float)
    node_values = np.array([3.0, -2.0, 1.5, -1.0, 4.5, -3.5, 2.2, -4.1], dtype=float)
    res1 = fcn(node_coords, node_values, num_gauss_pts=1)
    res4 = fcn(node_coords, node_values, num_gauss_pts=4)
    res9 = fcn(node_coords, node_values, num_gauss_pts=9)
    assert res1.shape == (2,)
    assert res4.shape == (2,)
    assert res9.shape == (2,)
    assert not np.allclose(res9, res1, rtol=1e-10, atol=1e-12)
    assert not np.allclose(res9, res4, rtol=1e-10, atol=1e-12)