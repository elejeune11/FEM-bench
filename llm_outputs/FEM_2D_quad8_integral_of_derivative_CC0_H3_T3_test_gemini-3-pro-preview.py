def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """
    Validate ∫_Ω (∇u) dΩ for a cubic field on an identity-mapped Q8 element.
    A 2×2 Gauss–Legendre rule (num_gauss_pts = 4) is exact for this case.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    node_values = node_coords[:, 0] ** 3 + node_coords[:, 1] ** 3
    result = fcn(node_coords, node_values, num_gauss_pts=4)
    expected = np.array([4.0, 4.0])
    assert np.allclose(result, expected, atol=1e-12), f'Expected {expected}, got {result}'

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """
    Analytic check with an affine geometric map and a linear field.
    Map the reference square Q = [-1, 1] × [-1, 1] by [x, y]^T = A[ξ, η]^T + c.
    For the linear scalar field u(x, y) = α + βx + γy.
    Test to make sure the function matches the correct analytical solution.
    """
    ref_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    phys_coords = np.zeros_like(ref_coords)
    phys_coords[:, 0] = 2.0 * ref_coords[:, 0] + 5.0
    phys_coords[:, 1] = 0.5 * ref_coords[:, 1] - 2.0
    (alpha, beta, gamma) = (10.0, 3.0, -7.0)
    node_values = alpha + beta * phys_coords[:, 0] + gamma * phys_coords[:, 1]
    result = fcn(phys_coords, node_values, num_gauss_pts=4)
    expected = np.array([12.0, -28.0])
    assert np.allclose(result, expected, atol=1e-12), f'Expected {expected}, got {result}'

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """
    Check quadrature-order sensitivity on a curved, asymmetric Q8 mapping.
    Select a geometry that is intentionally non-affine. With properly selected fixed,
    asymmetric nodal values, the integral ∫_Ω (∇u) dΩ should depend on the
    quadrature order. The test confirms that results from a 3×3 rule differ from those 
    of 1×1 or 2×2, verifying that higher-order integration is sensitive to nonlinear mappings.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    node_coords[5] = [1.5, 0.2]
    node_values = np.array([0.0, 1.0, 0.0, -1.0, 0.5, 0.5, -0.5, -0.5])
    res_1 = fcn(node_coords, node_values, num_gauss_pts=1)
    res_4 = fcn(node_coords, node_values, num_gauss_pts=4)
    res_9 = fcn(node_coords, node_values, num_gauss_pts=9)
    assert not np.allclose(res_9, res_1, rtol=0.01), '1-point result should differ from 9-point result on curved element'
    assert not np.allclose(res_9, res_4, rtol=1e-05), '4-point result should differ from 9-point result on curved element'