def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """Analytic check with identity mapping (reference element = physical element).
    When x ≡ ξ and y ≡ η, the Jacobian is J = I with det(J) = 1, so the routine
    integrates the physical gradient directly over Q = [-1, 1] × [-1, 1].
    Using u(x, y) = x^3 + y^3 gives ∇u = [3x^2, 3y^2] and the exact integrals on Q are [4, 4].
    A 2×2 Gauss rule (num_gauss_pts=4) is exact for this case; the test checks we recover [4, 4]."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    node_values = node_coords[:, 0] ** 3 + node_coords[:, 1] ** 3
    integral = fcn(node_coords, node_values, num_gauss_pts=4)
    assert np.allclose(integral, [4.0, 4.0])

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """Analytic check with an affine geometric map and a linear field. We map the
    reference square Q = [-1, 1] × [-1, 1] by [x, y]^T = A[ξ, η]^T + c. By placing
    the eight node quad mid-edge nodes at the arithmetic midpoints of the mapped corners, the
    isoparametric geometry is exactly affine, so the Jacobian is constant (J = A,
    det(J) = det(A)). For the linear scalar field u(x, y) = α + βx + γy, the physical
    gradient is constant ∇u = [β, γ], hence ∫_Ω ∇u dΩ = [β, γ] · Area(Ω). The area follows
    from the mapping: Area(Ω) = ∫_Q det(J) dQ = det(A) · Area(Q) = 4 · |det(A)|, so
    the exact result is [β, γ] · (4 · |det(A)|). Test to make sure the function matches
    this analytical solution."""
    A = np.array([[2.0, 0.5], [0.5, 1.0]])
    c = np.array([1.0, 2.0])
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    mapped_corners = corners @ A.T + c
    mid_edge_nodes = (mapped_corners[[0, 1, 2, 3, 0]].reshape(-1, 1, 2) + mapped_corners[[1, 2, 3, 0, 0]].reshape(-1, 1, 2)) / 2
    node_coords = np.vstack([mapped_corners, mid_edge_nodes.reshape(-1, 2)])
    alpha = 1.0
    beta = 2.0
    gamma = 3.0
    node_values = alpha + node_coords[:, 0] * beta + node_coords[:, 1] * gamma
    detA = np.linalg.det(A)
    exact_integral = np.array([beta, gamma]) * 4 * np.abs(detA)
    integral = fcn(node_coords, node_values, num_gauss_pts=4)
    assert np.allclose(integral, exact_integral)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """Test quadrature-order sensitivity on a deliberately curved, asymmetric mapping.
    One approach is to keep the four corners on the reference square but displace the mid-edge
    nodes asymmetrically, inducing a non-affine geometry (spatially varying J).
    With fixed, non-symmetric nodal values, the FE integrand becomes high-order in (ξ, η), 
    so a 3×3 rule should not coincide with 2×2 or 1×1.
    The test asserts that increasing the rule to 3×3 changes the result."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0.2, -1], [1, 0.3], [0, 1], [-1, -0.1]])
    node_values = np.arange(8)
    integral_4 = fcn(node_coords, node_values, num_gauss_pts=4)
    integral_9 = fcn(node_coords, node_values, num_gauss_pts=9)
    assert not np.allclose(integral_4, integral_9)