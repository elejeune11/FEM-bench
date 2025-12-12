def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """Validate ∫_Ω (∇u) dΩ for a cubic field on an identity-mapped Q8 element.
    A 2×2 Gauss–Legendre rule (num_gauss_pts = 4) is exact for this case,"""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    node_values = np.array([0, 1, 8, 7, 0.5, 3, 3.5, 2])
    num_gauss_pts = 4
    result = fcn(node_coords, node_values, num_gauss_pts)
    expected = np.array([12, 12])
    assert np.allclose(result, expected)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """Analytic check with an affine geometric map and a linear field.
    Map the reference square Q = [-1, 1] × [-1, 1] by [x, y]^T = A[ξ, η]^T + c.
    For the linear scalar field u(x, y) = α + βx + γy.
    Test to make sure the function matches the correct analytical solution."""
    A = np.array([[2, 0], [0, 3]])
    c = np.array([1, 2])
    node_coords = np.zeros((8, 2))
    for (i, (xi, eta)) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1), (0, -1), (1, 0), (0, 1), (-1, 0)]):
        node_coords[i] = A @ np.array([xi, eta]) + c
    (alpha, beta, gamma) = (1, 2, 3)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    num_gauss_pts = 1
    result = fcn(node_coords, node_values, num_gauss_pts)
    expected = np.array([beta, gamma]) * np.linalg.det(A) * 4
    assert np.allclose(result, expected)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """Check quadrature-order sensitivity on a curved, asymmetric Q8 mapping.
    Select a geometry that is intentionally non-affine. With properly selected fixed,
    asymmetric nodal values, the integral ∫_Ω (∇u) dΩ should depend on the
    quadrature order. The test confirms that results from a 3×3 rule differ from those 
    of 1×1 or 2×2, verifying that higher-order integration is sensitive to nonlinear mappings."""
    node_coords = np.array([[-1, -1.1], [1, -1], [1.2, 1.1], [-1, 1], [0, -1.05], [1.1, 0.1], [0.1, 1], [-1, 0.1]])
    node_values = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    result_1x1 = fcn(node_coords, node_values, 1)
    result_2x2 = fcn(node_coords, node_values, 4)
    result_3x3 = fcn(node_coords, node_values, 9)
    assert not np.allclose(result_1x1, result_3x3)
    assert not np.allclose(result_2x2, result_3x3)