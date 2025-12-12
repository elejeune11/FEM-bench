def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """
    The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised.
    """
    bad_inputs = [None, 42, 'invalid', [0.0, 0.0], np.array([0.0]), np.array([0.0, 0.0, 0.0]), np.array([[0.0], [1.0]]), np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, np.nan]]), np.array([np.inf, 0.0]), np.array([[0.0, 1.0], [np.inf, 0.0]])]
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
    points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]], dtype=float)
    (N, dN) = fcn(points)
    sums = N.sum(axis=(1, 2))
    assert np.allclose(sums, np.ones(points.shape[0]), atol=1e-13, rtol=0.0)

def test_derivative_partition_of_unity_quad8(fcn):
    """
    Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]], dtype=float)
    (_, dN) = fcn(points)
    grad_sums = dN.sum(axis=1)
    assert np.allclose(grad_sums, np.zeros_like(grad_sums), atol=1e-13, rtol=0.0)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """
    For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    (N, _) = fcn(nodes)
    N_mat = N.squeeze(-1)
    assert N_mat.shape == (8, 8)
    assert np.allclose(N_mat, np.eye(8), atol=1e-12, rtol=0.0)

def test_value_completeness_quad8(fcn):
    """
    Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    xi_nodes = nodes[:, 0]
    eta_nodes = nodes[:, 1]
    grid_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    pts = np.array([(x, y) for x in grid_vals for y in grid_vals], dtype=float)
    (N, _) = fcn(pts)
    Nm = N.squeeze(-1)
    polynomials = [lambda x, y: np.ones_like(x), lambda x, y: x, lambda x, y: y, lambda x, y: x * y, lambda x, y: x ** 2, lambda x, y: y ** 2, lambda x, y: 2.0 + 0.5 * x - 0.3 * y + 0.2 * x * y + 1.1 * x ** 2 - 0.7 * y ** 2]
    for poly in polynomials:
        u_nodes = poly(xi_nodes, eta_nodes)
        u_pred = Nm @ u_nodes
        u_exact = poly(pts[:, 0], pts[:, 1])
        assert np.allclose(u_pred, u_exact, atol=1e-12, rtol=0.0)

def test_gradient_completeness_quad8(fcn):
    """
    Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    xi_nodes = nodes[:, 0]
    eta_nodes = nodes[:, 1]
    grid_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    pts = np.array([(x, y) for x in grid_vals for y in grid_vals], dtype=float)
    (_, dN) = fcn(pts)
    polynomials = [(lambda x, y: np.ones_like(x), lambda x, y: np.stack((np.zeros_like(x), np.zeros_like(y)), axis=-1)), (lambda x, y: x, lambda x, y: np.stack((np.ones_like(x), np.zeros_like(y)), axis=-1)), (lambda x, y: y, lambda x, y: np.stack((np.zeros_like(x), np.ones_like(x)), axis=-1)), (lambda x, y: x * y, lambda x, y: np.stack((y, x), axis=-1)), (lambda x, y: x ** 2, lambda x, y: np.stack((2 * x, np.zeros_like(x)), axis=-1)), (lambda x, y: y ** 2, lambda x, y: np.stack((np.zeros_like(x), 2 * y), axis=-1)), (lambda x, y: 2.0 + 0.5 * x - 0.3 * y + 0.2 * x * y + 1.1 * x ** 2 - 0.7 * y ** 2, lambda x, y: np.stack((0.5 + 0.2 * y + 2.2 * x, -0.3 + 0.2 * x - 1.4 * y), axis=-1))]
    for (poly, grad_poly) in polynomials:
        u_nodes = poly(xi_nodes, eta_nodes)
        grad_rec = (dN * u_nodes[np.newaxis, :, np.newaxis]).sum(axis=1)
        grad_exact = grad_poly(pts[:, 0], pts[:, 1])
        assert np.allclose(grad_rec, grad_exact, atol=1e-12, rtol=0.0)