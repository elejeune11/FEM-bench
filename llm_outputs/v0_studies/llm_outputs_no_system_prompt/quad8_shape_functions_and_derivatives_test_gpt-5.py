def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """
    The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised.
    """
    bad_inputs = [[0.0, 0.0], (0.0, 0.0), np.array([0.0]), np.array([[0.0, 0.0, 0.0]]), np.array([[[0.0, 0.0]]]), np.array([np.nan, 0.0]), np.array([[np.inf, 0.0]])]
    for bad in bad_inputs:
        with pytest.raises(ValueError):
            fcn(bad)

def test_partition_of_unity_quad8(fcn):
    """
    Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance.
    """
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]], dtype=float)
    (N, _) = fcn(pts)
    Ns = N if N.ndim == 2 else N[..., 0]
    s = Ns.sum(axis=1)
    assert np.allclose(s, np.ones_like(s), atol=1e-12, rtol=0.0)

def test_derivative_partition_of_unity_quad8(fcn):
    """
    Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]], dtype=float)
    (_, dN) = fcn(pts)
    grad_sum = dN.sum(axis=1)
    zeros = np.zeros_like(grad_sum)
    assert np.allclose(grad_sum, zeros, atol=1e-12, rtol=0.0)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """
    For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    (N, _) = fcn(nodes)
    Ns = N if N.ndim == 2 else N[..., 0]
    A = Ns.T
    I = np.eye(8)
    assert np.allclose(A, I, atol=1e-12, rtol=0.0)

def test_value_completeness_quad8(fcn):
    """
    Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.3, -0.7], [-0.3, 0.2], [0.6, 0.4], [-0.6, -0.4], [0.8, 0.0], [0.0, -0.8]], dtype=float)
    a_lin = np.array([0.3, -1.2, 2.1], dtype=float)

    def p_lin(xi, eta):
        return a_lin[0] + a_lin[1] * xi + a_lin[2] * eta
    a_quad = np.array([0.7, -0.4, 0.9, 0.5, -0.8, 0.6], dtype=float)

    def p_quad(xi, eta):
        return a_quad[0] + a_quad[1] * xi + a_quad[2] * eta + a_quad[3] * xi ** 2 + a_quad[4] * xi * eta + a_quad[5] * eta ** 2
    u_nodes_lin = p_lin(nodes[:, 0], nodes[:, 1])
    u_nodes_quad = p_quad(nodes[:, 0], nodes[:, 1])
    (N_pts, _) = fcn(pts)
    Ns = N_pts if N_pts.ndim == 2 else N_pts[..., 0]
    u_lin_interp = Ns @ u_nodes_lin
    u_quad_interp = Ns @ u_nodes_quad
    u_lin_exact = p_lin(pts[:, 0], pts[:, 1])
    u_quad_exact = p_quad(pts[:, 0], pts[:, 1])
    err_lin = np.max(np.abs(u_lin_interp - u_lin_exact))
    err_quad = np.max(np.abs(u_quad_interp - u_quad_exact))
    assert err_lin <= 1e-12
    assert err_quad <= 1e-12

def test_gradient_completeness_quad8(fcn):
    """
    Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.3, -0.7], [-0.3, 0.2], [0.6, 0.4], [-0.6, -0.4], [0.8, 0.0], [0.0, -0.8]], dtype=float)
    a_lin = np.array([0.3, -1.2, 2.1], dtype=float)

    def grad_lin(xi, eta):
        return np.stack([np.full_like(xi, a_lin[1]), np.full_like(eta, a_lin[2])], axis=-1)
    a_quad = np.array([0.7, -0.4, 0.9, 0.5, -0.8, 0.6], dtype=float)

    def grad_quad(xi, eta):
        dxi = a_quad[1] + 2.0 * a_quad[3] * xi + a_quad[4] * eta
        deta = a_quad[2] + a_quad[4] * xi + 2.0 * a_quad[5] * eta
        return np.stack([dxi, deta], axis=-1)

    def p_lin(xi, eta):
        return a_lin[0] + a_lin[1] * xi + a_lin[2] * eta

    def p_quad(xi, eta):
        return a_quad[0] + a_quad[1] * xi + a_quad[2] * eta + a_quad[3] * xi ** 2 + a_quad[4] * xi * eta + a_quad[5] * eta ** 2
    u_nodes_lin = p_lin(nodes[:, 0], nodes[:, 1])
    u_nodes_quad = p_quad(nodes[:, 0], nodes[:, 1])
    (_, dN_pts) = fcn(pts)
    g_lin_rec = (dN_pts * u_nodes_lin[None, :, None]).sum(axis=1)
    g_quad_rec = (dN_pts * u_nodes_quad[None, :, None]).sum(axis=1)
    g_lin_exact = grad_lin(pts[:, 0], pts[:, 1])
    g_quad_exact = grad_quad(pts[:, 0], pts[:, 1])
    err_lin = np.max(np.linalg.norm(g_lin_rec - g_lin_exact, axis=1))
    err_quad = np.max(np.linalg.norm(g_quad_rec - g_quad_exact, axis=1))
    assert err_lin <= 1e-12
    assert err_quad <= 1e-12