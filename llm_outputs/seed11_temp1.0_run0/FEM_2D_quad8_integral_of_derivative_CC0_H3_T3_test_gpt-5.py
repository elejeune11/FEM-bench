def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """
    Validate ∫_Ω (∇u) dΩ for a cubic field on an identity-mapped Q8 element.
    A 2×2 Gauss–Legendre rule (num_gauss_pts = 4) is exact for this case,
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)

    def u(x, y):
        return x ** 3 + 2.0 * x ** 2 * y - y ** 3 + 3.0 * x * y
    node_values = np.array([u(x, y) for (x, y) in node_coords], dtype=float)
    expected = np.array([4.0, -4.0 / 3.0], dtype=float)
    integral = fcn(node_coords, node_values, num_gauss_pts=4)
    assert np.allclose(integral, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """
    Analytic check with an affine geometric map and a linear field.
    Map the reference square Q = [-1, 1] × [-1, 1] by [x, y]^T = A[ξ, η]^T + c.
    For the linear scalar field u(x, y) = α + βx + γy.
    Test to make sure the function matches the correct analytical solution.
    """
    ref_nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    A = np.array([[2.0, 0.5], [0.3, 1.5]], dtype=float)
    c = np.array([0.7, -1.2], dtype=float)
    node_coords = ref_nodes @ A.T + c
    (alpha, beta, gamma) = (0.4, -1.3, 2.1)

    def u(x, y):
        return alpha + beta * x + gamma * y
    node_values = np.array([u(x, y) for (x, y) in node_coords], dtype=float)
    area = np.linalg.det(A) * 4.0
    expected = np.array([beta * area, gamma * area], dtype=float)
    integral = fcn(node_coords, node_values, num_gauss_pts=1)
    assert np.allclose(integral, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """
    Check quadrature-order sensitivity on a curved, asymmetric Q8 mapping.
    Select a geometry that is intentionally non-affine. With properly selected fixed,
    asymmetric nodal values, the integral ∫_Ω (∇u) dΩ should depend on the
    quadrature order. The test confirms that results from a 3×3 rule differ from those 
    of 1×1 or 2×2, verifying that higher-order integration is sensitive to nonlinear mappings.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.1, -1.0], [1.0, 0.3], [-0.2, 1.0], [-1.0, -0.4]], dtype=float)
    node_values = np.array([0.5, -1.2, 0.8, -0.4, 1.1, -0.9, 0.7, -0.3], dtype=float)
    I1 = fcn(node_coords, node_values, num_gauss_pts=1)
    I4 = fcn(node_coords, node_values, num_gauss_pts=4)
    I9 = fcn(node_coords, node_values, num_gauss_pts=9)
    diff_91 = np.linalg.norm(I9 - I1)
    diff_94 = np.linalg.norm(I9 - I4)
    assert diff_91 > 1e-06
    assert diff_94 > 1e-06