def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """Analytic check with identity mapping (reference element = physical element). When x ≡ ξ and y ≡ η, the Jacobian is J = I with det(J) = 1, so the routine integrates the physical gradient directly over Q = [-1, 1] × [-1, 1]. Using u(x, y) = x^3 + y^3 gives ∇u = [3x^2, 3y^2] and the exact integrals on Q are [4, 4]. A 2×2 Gauss rule (num_gauss_pts=4) is exact for this case; the test checks we recover [4, 4]."""
    np = fcn.__globals__['np']
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    node_values = x ** 3 + y ** 3
    integral = fcn(node_coords, node_values, 4)
    expected = np.array([4.0, 4.0], dtype=float)
    assert integral.shape == (2,)
    assert np.allclose(integral, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """Analytic check with an affine geometric map and a linear field. We map the reference square Q = [-1, 1] × [-1, 1] by [x, y]^T = A[ξ, η]^T + c. By placing the eight node quad mid-edge nodes at the arithmetic midpoints of the mapped corners, the isoparametric geometry is exactly affine, so the Jacobian is constant (J = A, det(J) = det(A)). For the linear scalar field u(x, y) = α + βx + γy, the physical gradient is constant ∇u = [β, γ], hence ∫_Ω ∇u dΩ = [β, γ] · Area(Ω). The area follows from the mapping: Area(Ω) = ∫_Q det(J) dQ = det(A) · Area(Q) = 4 · |det(A)|, so the exact result is [β, γ] · (4 · |det(A)|). Test to make sure the function matches this analytical solution."""
    np = fcn.__globals__['np']
    xi_nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    A = np.array([[2.0, 0.5], [-0.3, 1.7]], dtype=float)
    c = np.array([0.7, -1.1], dtype=float)
    node_coords = xi_nodes @ A.T + c
    (alpha, beta, gamma) = (1.2, -0.8, 2.5)
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    node_values = alpha + beta * x + gamma * y
    detA = float(np.linalg.det(A))
    area = 4.0 * abs(detA)
    expected = np.array([beta * area, gamma * area], dtype=float)
    integral = fcn(node_coords, node_values, 4)
    assert integral.shape == (2,)
    assert np.allclose(integral, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """Test quadrature-order sensitivity on a deliberately curved, asymmetric mapping. One approach is to keep the four corners on the reference square but displace the mid-edge nodes asymmetrically, inducing a non-affine geometry (spatially varying J). With fixed, non-symmetric nodal values, the FE integrand becomes high-order in (ξ, η), so a 3×3 rule should not coincide with 2×2 or 1×1. The test asserts that increasing the rule to 3×3 changes the result."""
    np = fcn.__globals__['np']
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.3, -1.2], [1.2, 0.1], [-0.2, 1.1], [-1.1, -0.2]], dtype=float)
    node_values = np.array([0.7, -1.3, 2.1, 0.4, 1.5, -0.6, 2.2, -0.9], dtype=float)
    i1 = fcn(node_coords, node_values, 1)
    i4 = fcn(node_coords, node_values, 4)
    i9 = fcn(node_coords, node_values, 9)
    assert i1.shape == (2,)
    assert i4.shape == (2,)
    assert i9.shape == (2,)
    assert not np.allclose(i9, i4, rtol=1e-05, atol=1e-08)
    assert not np.allclose(i9, i1, rtol=1e-05, atol=1e-08)