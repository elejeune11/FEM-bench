def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """
    The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised.
    """
    bad_inputs = [[0.0, 0.0], (0.0, 0.0), '0,0', np.array(0.0), np.array([0.0, 0.0, 0.0]), np.array([[0.0, 0.0, 0.0]]), np.array([[0.0], [0.0]]), np.array([[[0.0, 0.0]]]), np.array([np.nan, 0.0]), np.array([np.inf, 0.0]), np.array([[0.0, np.nan]]), np.array([[0.0, 0.0], [np.inf, -np.inf]])]
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
    N, dN = fcn(pts)
    Nsum = np.squeeze(N, axis=-1).sum(axis=1)
    assert np.allclose(Nsum, np.ones(pts.shape[0]), rtol=0.0, atol=1e-14)

def test_derivative_partition_of_unity_quad8(fcn):
    """
    Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    _, dN = fcn(pts)
    dNsum = dN.sum(axis=1)
    assert np.allclose(dNsum, np.zeros((pts.shape[0], 2)), rtol=0.0, atol=1e-14)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """
    For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    N, _ = fcn(nodes)
    M = np.squeeze(N, axis=-1)
    assert np.allclose(M, np.eye(8), rtol=0.0, atol=1e-14)

def test_value_completeness_quad8(fcn):
    """
    Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    g = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    Xi, Eta = np.meshgrid(g, g, indexing='xy')
    pts = np.column_stack([Xi.ravel(), Eta.ravel()])
    a0, a1, a2 = (3.2, 1.1, -0.7)
    p1 = lambda x, y: a0 + a1 * x + a2 * y
    b00, b10, b01, b20, b11, b02 = (0.3, -1.7, 0.9, 2.3, -0.4, 1.25)
    p2 = lambda x, y: b00 + b10 * x + b01 * y + b20 * x * x + b11 * x * y + b02 * y * y
    V1 = p1(nodes[:, 0], nodes[:, 1])
    N_pts, _ = fcn(pts)
    N2d = np.squeeze(N_pts, axis=-1)
    interp1 = N2d @ V1
    truth1 = p1(pts[:, 0], pts[:, 1])
    err1 = np.max(np.abs(interp1 - truth1))
    V2 = p2(nodes[:, 0], nodes[:, 1])
    interp2 = N2d @ V2
    truth2 = p2(pts[:, 0], pts[:, 1])
    err2 = np.max(np.abs(interp2 - truth2))
    assert err1 < 1e-13
    assert err2 < 1e-12

def test_gradient_completeness_quad8(fcn):
    """
    Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    g = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    Xi, Eta = np.meshgrid(g, g, indexing='xy')
    pts = np.column_stack([Xi.ravel(), Eta.ravel()])
    a0, a1, a2 = (3.2, 1.1, -0.7)
    p1 = lambda x, y: a0 + a1 * x + a2 * y
    grad_p1 = lambda x, y: np.column_stack([np.full_like(x, a1), np.full_like(y, a2)])
    b00, b10, b01, b20, b11, b02 = (0.3, -1.7, 0.9, 2.3, -0.4, 1.25)
    p2 = lambda x, y: b00 + b10 * x + b01 * y + b20 * x * x + b11 * x * y + b02 * y * y
    grad_p2 = lambda x, y: np.column_stack([b10 + 2.0 * b20 * x + b11 * y, b01 + 2.0 * b02 * y + b11 * x])
    V1 = p1(nodes[:, 0], nodes[:, 1])
    V2 = p2(nodes[:, 0], nodes[:, 1])
    _, dN = fcn(pts)
    grad1 = np.einsum('i,nij->nj', V1, dN)
    grad2 = np.einsum('i,nij->nj', V2, dN)
    true_grad1 = grad_p1(pts[:, 0], pts[:, 1])
    true_grad2 = grad_p2(pts[:, 0], pts[:, 1])
    err1 = np.max(np.abs(grad1 - true_grad1))
    err2 = np.max(np.abs(grad2 - true_grad2))
    assert err1 < 1e-13
    assert err2 < 1e-12