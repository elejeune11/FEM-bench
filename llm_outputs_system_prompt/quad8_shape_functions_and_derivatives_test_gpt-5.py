def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    bad_inputs = [None, 0, 3.14, 'bad', [0.0, 0.0], (0.0, 0.0), np.array([0.0]), np.array([[0.0, 0.0, 0.0]]), np.array([[[0.0, 0.0]]]), np.array([np.nan, 0.0]), np.array([[0.0, np.inf]])]
    for x in bad_inputs:
        try:
            _ = fcn(x)
        except ValueError:
            pass
        else:
            assert False, f'Expected ValueError for invalid input: {x!r}'

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    (N, dN) = fcn(pts)
    s = N[:, :, 0].sum(axis=1)
    assert np.allclose(s, np.ones_like(s), atol=1e-12, rtol=0.0)

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    (N, dN) = fcn(pts)
    gsum = dN.sum(axis=1)
    assert np.allclose(gsum, np.zeros_like(gsum), atol=1e-12, rtol=0.0)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (N, dN) = fcn(nodes)
    mat = N[:, :, 0]
    I = np.eye(8)
    assert np.allclose(mat, I, atol=1e-12, rtol=0.0)

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    xi = np.linspace(-1.0, 1.0, 5)
    eta = np.linspace(-1.0, 1.0, 5)
    (X, Y) = np.meshgrid(xi, eta, indexing='xy')
    pts = np.column_stack([X.ravel(), Y.ravel()])
    (N_pts, _) = fcn(pts)
    (c0L, c1L, c2L) = (0.7, -1.2, 2.3)
    vL = c0L + c1L * nodes[:, 0] + c2L * nodes[:, 1]
    predL = N_pts[:, :, 0] @ vL
    exactL = c0L + c1L * pts[:, 0] + c2L * pts[:, 1]
    errL = np.max(np.abs(predL - exactL))
    assert errL <= 1e-12
    (c0Q, c1Q, c2Q, c3Q, c4Q, c5Q) = (0.3, -0.9, 1.4, 0.8, -0.6, 0.5)
    vQ = c0Q + c1Q * nodes[:, 0] + c2Q * nodes[:, 1] + c3Q * nodes[:, 0] * nodes[:, 1] + c4Q * nodes[:, 0] ** 2 + c5Q * nodes[:, 1] ** 2
    predQ = N_pts[:, :, 0] @ vQ
    exactQ = c0Q + c1Q * pts[:, 0] + c2Q * pts[:, 1] + c3Q * pts[:, 0] * pts[:, 1] + c4Q * pts[:, 0] ** 2 + c5Q * pts[:, 1] ** 2
    errQ = np.max(np.abs(predQ - exactQ))
    assert errQ <= 1e-12

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    xi = np.linspace(-1.0, 1.0, 5)
    eta = np.linspace(-1.0, 1.0, 5)
    (X, Y) = np.meshgrid(xi, eta, indexing='xy')
    pts = np.column_stack([X.ravel(), Y.ravel()])
    (_, dN_pts) = fcn(pts)
    (c0L, c1L, c2L) = (0.7, -1.2, 2.3)
    vL = c0L + c1L * nodes[:, 0] + c2L * nodes[:, 1]
    grad_pred_L = (dN_pts * vL[None, :, None]).sum(axis=1)
    grad_exact_L = np.column_stack([np.full(pts.shape[0], c1L), np.full(pts.shape[0], c2L)])
    errL = np.max(np.abs(grad_pred_L - grad_exact_L))
    assert errL <= 1e-12
    (c0Q, c1Q, c2Q, c3Q, c4Q, c5Q) = (0.3, -0.9, 1.4, 0.8, -0.6, 0.5)
    vQ = c0Q + c1Q * nodes[:, 0] + c2Q * nodes[:, 1] + c3Q * nodes[:, 0] * nodes[:, 1] + c4Q * nodes[:, 0] ** 2 + c5Q * nodes[:, 1] ** 2
    grad_pred_Q = (dN_pts * vQ[None, :, None]).sum(axis=1)
    dphidx = c1Q + c3Q * pts[:, 1] + 2.0 * c4Q * pts[:, 0]
    dphideta = c2Q + c3Q * pts[:, 0] + 2.0 * c5Q * pts[:, 1]
    grad_exact_Q = np.column_stack([dphidx, dphideta])
    errQ = np.max(np.abs(grad_pred_Q - grad_exact_Q))
    assert errQ <= 1e-12