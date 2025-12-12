def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised."""
    with pytest.raises(ValueError):
        fcn([0.0, 0.0])
    with pytest.raises(ValueError):
        fcn(np.array([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0], [1.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, 0.0], [np.inf, 0.0]]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures
    that the sum equals 1 within tight tolerance."""
    pts = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1 / 3, 1 / 3], [0.2, 0.3], [0.6, 0.1], [0.1, 0.2]])
    (N, dN) = fcn(pts)
    Nmat = np.squeeze(N, axis=-1)
    totals = np.sum(Nmat, axis=1)
    assert np.allclose(totals, 1.0, atol=1e-12, rtol=0)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    pts = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.25, 0.25], [0.3, 0.4]])
    (N, dN) = fcn(pts)
    sum_grad = np.sum(dN, axis=1)
    assert np.allclose(sum_grad, 0.0, atol=1e-12, rtol=0)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N, dN) = fcn(nodes)
    Nmat = np.squeeze(N, axis=-1)
    A = Nmat.T
    assert np.allclose(A, np.eye(6), atol=1e-12, rtol=0)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    pts = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1 / 3, 1 / 3], [0.2, 0.3], [0.6, 0.1], [0.1, 0.2]])
    polys = [lambda x, y: 2.0 + 0.0 * x + 0.0 * y, lambda x, y: x, lambda x, y: y, lambda x, y: x * x, lambda x, y: x * y, lambda x, y: y * y, lambda x, y: 1.3 + 2.2 * x - 1.7 * y + 0.5 * x * x + 0.8 * x * y - 0.6 * y * y]
    (N_nodes, _) = fcn(nodes)
    N_nodes = np.squeeze(N_nodes, axis=-1)
    (N_pts, _) = fcn(pts)
    N_pts = np.squeeze(N_pts, axis=-1)
    tol = 1e-12
    for p in polys:
        u_nodes = p(nodes[:, 0], nodes[:, 1])
        u_nodes = np.asarray(u_nodes).reshape(6)
        u_interp = N_pts.dot(u_nodes)
        u_exact = p(pts[:, 0], pts[:, 1])
        assert np.allclose(u_interp, u_exact, atol=tol, rtol=0)

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    pts = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1 / 3, 1 / 3], [0.2, 0.3], [0.6, 0.1], [0.1, 0.2]])
    (N_nodes, _) = fcn(nodes)
    N_nodes = np.squeeze(N_nodes, axis=-1)
    (N_pts, dN_pts) = fcn(pts)
    polys = [(lambda x, y: np.ones_like(x) * 3.7, lambda x, y: (np.zeros_like(x), np.zeros_like(x))), (lambda x, y: x, lambda x, y: (np.ones_like(x), np.zeros_like(x))), (lambda x, y: y, lambda x, y: (np.zeros_like(x), np.ones_like(x))), (lambda x, y: x * x, lambda x, y: (2 * x, np.zeros_like(x))), (lambda x, y: y * y, lambda x, y: (np.zeros_like(x), 2 * y)), (lambda x, y: x * y, lambda x, y: (y, x)), (lambda x, y: 1.3 + 2.2 * x - 1.7 * y + 0.5 * x * x + 0.8 * x * y - 0.6 * y * y, lambda x, y: (2 * 0.5 * x + 2.2 + 0.8 * y, 2 * -0.6 * y - 1.7 + 0.8 * x))]
    tol = 1e-12
    for (p, gradp) in polys:
        u_nodes = p(nodes[:, 0], nodes[:, 1]).reshape(6)
        grad_recon = np.einsum('i,nid->nd', u_nodes, dN_pts)
        (gx_exact, gy_exact) = gradp(pts[:, 0], pts[:, 1])
        grad_exact = np.vstack([gx_exact, gy_exact]).T
        assert np.allclose(grad_recon, grad_exact, atol=tol, rtol=0)