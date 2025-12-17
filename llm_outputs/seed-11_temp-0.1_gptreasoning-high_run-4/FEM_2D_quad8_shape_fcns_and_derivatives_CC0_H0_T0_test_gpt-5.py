def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """
    The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised.
    """
    bad_inputs = [3.14, 'not-an-array', {'xi': 0}, [0.0, 0.0], (0.0, 0.0), np.array([0.0]), np.array([0.0, 0.0, 0.0]), np.array([[0.0, 0.0, 0.0]]), np.zeros((2, 1)), np.zeros((3, 2, 1)), np.array([np.nan, 0.0]), np.array([[0.5, np.inf]])]
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
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    N, _ = fcn(pts)
    s = N.sum(axis=(1, 2))
    assert np.allclose(s, np.ones_like(s), atol=1e-12)

def test_derivative_partition_of_unity_quad8(fcn):
    """
    Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    _, dN = fcn(pts)
    grad_sum = dN.sum(axis=1)
    assert np.allclose(grad_sum, np.zeros_like(grad_sum), atol=1e-12)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """
    For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    N, _ = fcn(nodes)
    M = N[:, :, 0]
    I = np.eye(8)
    assert M.shape == (8, 8)
    assert np.allclose(M, I, atol=1e-12)

def test_value_completeness_quad8(fcn):
    """
    Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.3, -0.8], [-0.6, 0.4], [0.75, -0.25], [-0.25, 0.75]])

    def eval_poly_list(points, polys):
        xi = points[:, 0]
        eta = points[:, 1]
        return [p(xi, eta) for p in polys]
    polys = [lambda x, y: x, lambda x, y: y, lambda x, y: 1.2 + 0.8 * x - 1.6 * y, lambda x, y: x * x, lambda x, y: y * y, lambda x, y: x * y, lambda x, y: 0.25 + 0.7 * x - 1.1 * y + 0.3 * x * x - 0.9 * x * y + 1.2 * y * y]
    N_pts, _ = fcn(pts)
    N_pts = N_pts[:, :, 0]
    max_err = 0.0
    for p in polys:
        u_nodes = p(nodes[:, 0], nodes[:, 1])
        u_interp = N_pts @ u_nodes
        u_exact = p(pts[:, 0], pts[:, 1])
        max_err = max(max_err, float(np.max(np.abs(u_interp - u_exact))))
    assert max_err < 1e-12

def test_gradient_completeness_quad8(fcn):
    """
    Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.3, -0.8], [-0.6, 0.4], [0.75, -0.25], [-0.25, 0.75]])

    def p_lin(x, y):
        return 1.2 + 0.8 * x - 1.6 * y

    def grad_p_lin(x, y):
        gx = 0.8 + 0.0 * x
        gy = -1.6 + 0.0 * y
        return np.column_stack([gx, gy])
    a0, a1, a2, a3, a4, a5 = (0.25, 0.7, -1.1, 0.3, -0.9, 1.2)

    def p_quad(x, y):
        return a0 + a1 * x + a2 * y + a3 * x * x + a4 * x * y + a5 * y * y

    def grad_p_quad(x, y):
        gx = a1 + 2.0 * a3 * x + a4 * y
        gy = a2 + a4 * x + 2.0 * a5 * y
        return np.column_stack([gx, gy])
    polys = [(p_lin, grad_p_lin), (p_quad, grad_p_quad)]
    max_err = 0.0
    for p, grad_p in polys:
        u_nodes = p(nodes[:, 0], nodes[:, 1])
        _, dN_pts = fcn(pts)
        grad_interp = (dN_pts * u_nodes[None, :, None]).sum(axis=1)
        grad_exact = grad_p(pts[:, 0], pts[:, 1])
        max_err = max(max_err, float(np.max(np.abs(grad_interp - grad_exact))))
    assert max_err < 1e-12