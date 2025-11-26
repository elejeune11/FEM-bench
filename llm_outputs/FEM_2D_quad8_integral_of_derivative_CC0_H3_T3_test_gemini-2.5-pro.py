def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """Validate ∫_Ω (∇u) dΩ for a cubic field on an identity-mapped Q8 element.
A 2×2 Gauss–Legendre rule (num_gauss_pts = 4) is exact for this case,"""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (x, y) = (node_coords[:, 0], node_coords[:, 1])
    node_values = x ** 3 + 2 * y ** 3
    expected_integral = np.array([4.0, 8.0])
    num_gauss_pts = 4
    computed_integral = fcn(node_coords, node_values, num_gauss_pts)
    assert computed_integral.shape == (2,)
    assert np.allclose(computed_integral, expected_integral)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """Analytic check with an affine geometric map and a linear field.
Map the reference square Q = [-1, 1] × [-1, 1] by [x, y]^T = A[ξ, η]^T + c.
For the linear scalar field u(x, y) = α + βx + γy.
Test to make sure the function matches the correct analytical solution."""
    ref_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    A = np.array([[2.0, 1.0], [0.5, 3.0]])
    c = np.array([5.0, -2.0])
    node_coords = (A @ ref_coords.T).T + c
    det_J = np.linalg.det(A)
    area = 4.0 * det_J
    (beta, gamma) = (2.5, -1.5)
    node_values = beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    grad_u = np.array([beta, gamma])
    expected_integral = grad_u * area
    num_gauss_pts = 4
    computed_integral = fcn(node_coords, node_values, num_gauss_pts)
    assert computed_integral.shape == (2,)
    assert np.allclose(computed_integral, expected_integral)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """Check quadrature-order sensitivity on a curved, asymmetric Q8 mapping.
Select a geometry that is intentionally non-affine. With properly selected fixed,
asymmetric nodal values, the integral ∫_Ω (∇u) dΩ should depend on the
quadrature order. The test confirms that results from a 3×3 rule differ from those 
of 1×1 or 2×2, verifying that higher-order integration is sensitive to nonlinear mappings."""
    node_coords = np.array([[0.0, 0.0], [2.0, 0.2], [2.1, 2.0], [0.3, 2.0], [1.0, -0.1], [2.2, 1.1], [1.0, 2.5], [0.1, 1.0]])
    node_values = np.array([1.0, 2.5, 5.0, 3.0, 1.5, 4.0, 6.0, 2.0])
    integral_1pt = fcn(node_coords, node_values, num_gauss_pts=1)
    integral_4pt = fcn(node_coords, node_values, num_gauss_pts=4)
    integral_9pt = fcn(node_coords, node_values, num_gauss_pts=9)
    assert not np.allclose(integral_1pt, integral_4pt)
    assert not np.allclose(integral_4pt, integral_9pt)