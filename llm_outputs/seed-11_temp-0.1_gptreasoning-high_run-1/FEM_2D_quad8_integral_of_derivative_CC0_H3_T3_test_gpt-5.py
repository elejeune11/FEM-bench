def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """
    Validate ∫_Ω (∇u) dΩ for a cubic field on an identity-mapped Q8 element.
    Use u(x,y) = (x^3 - x) + (y^3 - y), which vanishes at all Q8 nodes.
    With num_gauss_pts = 4 (2×2 Gauss–Legendre), the integral should be [0, 0].
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)

    def u(x, y):
        return x ** 3 - x + (y ** 3 - y)
    node_values = np.array([u(x, y) for x, y in node_coords], dtype=float)
    assert np.allclose(node_values, 0.0)
    integral = fcn(node_coords, node_values, num_gauss_pts=4)
    assert isinstance(integral, np.ndarray) and integral.shape == (2,)
    assert np.allclose(integral, np.zeros(2), rtol=1e-13, atol=1e-13)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """
    Analytic check with an affine geometric map and a linear field.
    Map Q = [-1,1]^2 by [x, y]^T = A[ξ, η]^T + c.
    For u(x, y) = α + βx + γy, ∫_Ω ∇u dΩ = [β, γ] * area(Ω) where area(Ω) = 4 * det(A).
    Verify results for 1×1, 2×2, and 3×3 Gauss rules.
    """
    xi_eta = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    A = np.array([[1.8, 0.2], [0.5, 1.3]], dtype=float)
    c = np.array([0.7, -0.6], dtype=float)
    node_coords = xi_eta @ A.T + c
    alpha, beta, gamma = (0.3, 0.7, -1.1)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    detA = np.linalg.det(A)
    area = 4.0 * detA
    expected = np.array([beta * area, gamma * area], dtype=float)
    for ng in (1, 4, 9):
        integral = fcn(node_coords, node_values, num_gauss_pts=ng)
        assert isinstance(integral, np.ndarray) and integral.shape == (2,)
        assert np.allclose(integral, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """
    Check quadrature-order sensitivity on a curved, asymmetric Q8 mapping.
    Use a non-affine geometry and asymmetric nodal values. Expect 1×1 to differ
    from higher-order results, while 2×2 and 3×3 should agree for Q8 integrands.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.2, -1.0], [1.0, 0.3], [-0.25, 1.0], [-1.0, -0.15]], dtype=float)
    node_values = np.array([0.7, -0.3, 1.1, -0.4, 0.9, -1.2, 0.5, -0.6], dtype=float)
    I1 = fcn(node_coords, node_values, num_gauss_pts=1)
    I4 = fcn(node_coords, node_values, num_gauss_pts=4)
    I9 = fcn(node_coords, node_values, num_gauss_pts=9)
    assert I1.shape == (2,) and I4.shape == (2,) and (I9.shape == (2,))
    diff_1_9 = np.linalg.norm(I1 - I9, ord=2)
    assert diff_1_9 > 1e-08
    assert np.allclose(I4, I9, rtol=1e-12, atol=1e-12)