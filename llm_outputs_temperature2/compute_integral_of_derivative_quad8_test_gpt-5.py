def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """
    Analytic check with identity mapping and cubic field u(x,y) = x^3 + y^3 on Q = [-1,1]^2.
    The integral of the gradient is [4, 4] using a 2x2 Gauss rule (num_gauss_pts=4).
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    node_values = x ** 3 + y ** 3
    integral = fcn(node_coords, node_values, num_gauss_pts=4)
    expected = np.array([4.0, 4.0])
    assert integral.shape == (2,)
    assert np.allclose(integral, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """
    Analytic check for an affine geometric map and linear field u(x,y)=α+βx+γy.
    With x = A[ξ,η]^T + c and nodes placed by mapping the reference nodes through A and c,
    the Jacobian is constant and ∫Ω ∇u dΩ = [β, γ] * (4 * |det(A)|).
    """
    ref_nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    A = np.array([[1.7, -0.3], [0.8, 2.1]])
    c = np.array([0.3, -1.2])
    detA = np.linalg.det(A)
    node_coords = ref_nodes.dot(A.T) + c
    alpha = 0.25
    beta = -1.1
    gamma = 0.7
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    node_values = alpha + beta * x + gamma * y
    integral = fcn(node_coords, node_values, num_gauss_pts=1)
    expected = np.array([beta, gamma]) * (4.0 * abs(detA))
    assert integral.shape == (2,)
    assert np.allclose(integral, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """
    Order sensitivity check on a curved, asymmetric geometry. Corners fixed on the square,
    midside nodes displaced asymmetrically. With nonsymmetric nodal values, the integrand
    is higher-order, so 3x3 quadrature should yield a different result than 2x2 or 1x1.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.3, -0.8], [1.2, -0.1], [-0.2, 1.3], [-1.1, -0.4]])
    node_values = np.array([-0.7, 1.2, 0.9, -0.1, 2.3, -1.6, 0.4, 0.8])
    integral_1 = fcn(node_coords, node_values, num_gauss_pts=1)
    integral_4 = fcn(node_coords, node_values, num_gauss_pts=4)
    integral_9 = fcn(node_coords, node_values, num_gauss_pts=9)
    assert not np.allclose(integral_9, integral_4, rtol=1e-09, atol=1e-12)
    assert not np.allclose(integral_9, integral_1, rtol=1e-09, atol=1e-12)