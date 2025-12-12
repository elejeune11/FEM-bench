def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """
    Validate ∫_Ω (∇u) dΩ for a cubic field on an identity-mapped Q8 element.
    A 2×2 Gauss–Legendre rule (num_gauss_pts = 4) is exact for this case,
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)

    def u_fun(x, y):
        return x ** 3 + 2 * x ** 2 * y - x * y ** 2 + 5 * y ** 3 + 4 * x + 3 * y + 7
    node_values = np.array([u_fun(x, y) for (x, y) in node_coords], dtype=float)
    expected = np.array([56.0 / 3.0, 104.0 / 3.0], dtype=float)
    result = fcn(node_coords, node_values, 4)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """
    Analytic check with an affine geometric map and a linear field.
    Map the reference square Q = [-1, 1] × [-1, 1] by [x, y]^T = A[ξ, η]^T + c.
    For the linear scalar field u(x, y) = α + βx + γy.
    Test to make sure the function matches the correct analytical solution.
    """
    ref_nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    A = np.array([[1.2, 0.3], [0.4, 1.1]], dtype=float)
    c = np.array([0.5, -0.6], dtype=float)
    node_coords = ref_nodes @ A.T + c
    alpha = -0.2
    beta = 2.5
    gamma = -1.7
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    area = 4.0 * np.linalg.det(A)
    expected = np.array([beta * area, gamma * area], dtype=float)
    for ngp in (1, 4, 9):
        result = fcn(node_coords, node_values, ngp)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """
    Check quadrature-order sensitivity on a curved, asymmetric Q8 mapping.
    Select a geometry that is intentionally non-affine. With properly selected fixed,
    asymmetric nodal values, the integral ∫_Ω (∇u) dΩ should depend on the
    quadrature order. The test confirms that results from a 3×3 rule differ from those 
    of 1×1 or 2×2, verifying that higher-order integration is sensitive to nonlinear mappings.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.1, -1.2], [1.15, 0.2], [0.3, 1.25], [-1.25, -0.1]], dtype=float)
    node_values = np.array([-0.7, 2.3, -1.9, 0.4, 3.1, -2.6, 1.8, -0.2], dtype=float)
    I1 = fcn(node_coords, node_values, 1)
    I4 = fcn(node_coords, node_values, 4)
    I9 = fcn(node_coords, node_values, 9)
    assert isinstance(I1, np.ndarray) and I1.shape == (2,)
    assert isinstance(I4, np.ndarray) and I4.shape == (2,)
    assert isinstance(I9, np.ndarray) and I9.shape == (2,)
    assert not np.allclose(I9, I1, rtol=1e-08, atol=1e-10)
    assert np.allclose(I9, I4, rtol=1e-10, atol=1e-12)