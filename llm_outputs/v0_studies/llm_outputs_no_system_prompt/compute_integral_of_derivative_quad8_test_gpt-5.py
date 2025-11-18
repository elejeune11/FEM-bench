def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """
    Analytic check with identity mapping (reference element = physical element).
    Using u(x, y) = x^3 + y^3 on Q = [-1, 1]^2 gives exact integrals [4, 4] for ∫∇u dΩ.
    2×2 Gauss rule (num_gauss_pts=4) should recover this exactly.
    """
    ref_nodes = np.array([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)], dtype=float)
    node_coords = ref_nodes.copy()
    node_values = np.array([x ** 3 + y ** 3 for (x, y) in node_coords], dtype=float)
    res = fcn(node_coords, node_values, num_gauss_pts=4)
    expected = np.array([4.0, 4.0], dtype=float)
    assert np.allclose(res, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """
    Analytic check with an affine geometric map and a linear field.
    With x = A[ξ,η]^T + c and u(x,y) = α + βx + γy, ∇u = [β, γ] is constant and
    ∫_Ω ∇u dΩ = [β, γ] · (4 · |det(A)|).
    """
    ref_nodes = np.array([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)], dtype=float)
    A = np.array([[2.0, 0.5], [0.3, 1.5]], dtype=float)
    c = np.array([1.2, -0.7], dtype=float)
    node_coords = ref_nodes @ A.T + c
    (alpha, beta, gamma) = (0.35, -1.25, 2.1)
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    node_values = alpha + beta * x + gamma * y
    detA = float(np.linalg.det(A))
    area = 4.0 * abs(detA)
    expected = np.array([beta * area, gamma * area], dtype=float)
    res = fcn(node_coords, node_values, num_gauss_pts=1)
    assert np.allclose(res, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """
    Test quadrature-order sensitivity on a deliberately curved, asymmetric mapping.
    Non-affine geometry with non-symmetric nodal values should yield different
    results when increasing the quadrature from 1×1 or 2×2 to 3×3.
    """
    node_coords = np.array([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (0.2, -0.9), (1.1, 0.3), (-0.4, 1.2), (-1.2, -0.1)], dtype=float)
    node_values = np.array([0.7, -1.2, 2.5, 0.3, -0.4, 1.7, -2.1, 0.9], dtype=float)
    res1 = fcn(node_coords, node_values, num_gauss_pts=1)
    res4 = fcn(node_coords, node_values, num_gauss_pts=4)
    res9 = fcn(node_coords, node_values, num_gauss_pts=9)
    assert not np.allclose(res9, res4, rtol=1e-08, atol=1e-10)
    assert not np.allclose(res9, res1, rtol=1e-08, atol=1e-10)