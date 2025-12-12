def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """Validate ∫_Ω (∇u) dΩ for a cubic field on an identity-mapped Q8 element.
    A 2×2 Gauss–Legendre rule (num_gauss_pts = 4) is exact for this case."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=float)
    node_values = np.array([(-1) ** 3 + (-1) ** 3, 1 ** 3 + (-1) ** 3, 1 ** 3 + 1 ** 3, (-1) ** 3 + 1 ** 3, 0 ** 3 + (-1) ** 3, 1 ** 3 + 0 ** 3, 0 ** 3 + 1 ** 3, (-1) ** 3 + 0 ** 3])
    result = fcn(node_coords, node_values, 4)
    expected = np.array([16 / 3, 16 / 3])
    np.testing.assert_allclose(result, expected, rtol=1e-10)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """Analytic check with an affine geometric map and a linear field.
    Map the reference square Q = [-1, 1] × [-1, 1] by [x, y]^T = A[ξ, η]^T + c.
    For the linear scalar field u(x, y) = α + βx + γy.
    Test to make sure the function matches the correct analytical solution."""
    A = np.array([[2, 3], [-1, 4]])
    c = np.array([5, 2])
    ref_nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    node_coords = ref_nodes @ A.T + c
    (alpha, beta, gamma) = (3, 2, 5)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    result = fcn(node_coords, node_values, 1)
    expected = np.array([beta * 4, gamma * 4])
    np.testing.assert_allclose(result, expected, rtol=1e-12)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """Check quadrature-order sensitivity on a curved, asymmetric Q8 mapping.
    Select a geometry that is intentionally non-affine. With properly selected fixed,
    asymmetric nodal values, the integral ∫_Ω (∇u) dΩ should depend on the
    quadrature order. The test confirms that results from a 3×3 rule differ from those 
    of 1×1 or 2×2, verifying that higher-order integration is sensitive to nonlinear mappings."""
    node_coords = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, -0.1], [2.1, 1], [1, 2.1], [-0.1, 1]])
    node_values = np.array([1, 2, 3, 4, 1.5, 2.5, 3.5, 0.5])
    result_1 = fcn(node_coords, node_values, 1)
    result_2 = fcn(node_coords, node_values, 4)
    result_3 = fcn(node_coords, node_values, 9)
    assert not np.allclose(result_1, result_3, rtol=1e-06)
    assert not np.allclose(result_2, result_3, rtol=1e-06)