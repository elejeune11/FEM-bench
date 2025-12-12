def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """
    Validate ∫_Ω (∇u) dΩ for a cubic field on an identity-mapped Q8 element.
    A 2×2 Gauss–Legendre rule (num_gauss_pts = 4) is exact for this case.
    """
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    node_values = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    num_gauss_pts = 4
    result = fcn(node_coords, node_values, num_gauss_pts)
    assert np.allclose(result, np.array([0, 0]))

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """
    Analytic check with an affine geometric map and a linear field.
    Map the reference square Q = [-1, 1] × [-1, 1] by [x, y]^T = A[ξ, η]^T + c.
    For the linear scalar field u(x, y) = α + βx + γy.
    Test to make sure the function matches the correct analytical solution.
    """
    A = np.array([[2, 0], [0, 3]])
    c = np.array([1, 2])
    node_coords = A @ np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]]).T + c.T
    node_values = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    num_gauss_pts = 1
    result = fcn(node_coords.T, node_values, num_gauss_pts)
    assert np.allclose(result, np.array([0, 0]))

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """
    Check quadrature-order sensitivity on a curved, asymmetric Q8 mapping.
    Select a geometry that is intentionally non-affine. With properly selected fixed,
    asymmetric nodal values, the integral ∫_Ω (∇u) dΩ should depend on the
    quadrature order. The test confirms that results from a 3×3 rule differ from those 
    of 1×1 or 2×2, verifying that higher-order integration is sensitive to nonlinear mappings.
    """
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0.5, -1], [1, 0.5], [0, 1], [-1, 0.5]])
    node_values = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    result_1pt = fcn(node_coords, node_values, 1)
    result_4pt = fcn(node_coords, node_values, 4)
    result_9pt = fcn(node_coords, node_values, 9)
    assert not np.allclose(result_1pt, result_9pt)
    assert not np.allclose(result_4pt, result_9pt)