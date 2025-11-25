def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    with pytest.raises(ValueError):
        fcn([0.0, 0.0])
    with pytest.raises(ValueError):
        fcn(np.array([0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, np.inf]]))

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    sample_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (N, _) = fcn(sample_points)
    sums = np.sum(N, axis=1).flatten()
    assert np.allclose(sums, 1.0, atol=1e-14)

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    sample_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (_, dN_dxi) = fcn(sample_points)
    grad_sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(grad_sums, 0.0, atol=1e-14)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (N, _) = fcn(nodes)
    identity_matrix = np.eye(8)
    assert np.allclose(N.reshape(8, 8), identity_matrix, atol=1e-14)

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    sample_points = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0.0, -0.7], [0.7, 0.0], [0.0, 0.7], [-0.7, 0.0]])
    (N_nodes, _) = fcn(nodes)
    (N_samples, _) = fcn(sample_points)
    polynomials = [lambda x, y: 1.0, lambda x, y: x, lambda x, y: y, lambda x, y: x * y, lambda x, y: x ** 2, lambda x, y: y ** 2]
    for poly in polynomials:
        nodal_vals = np.array([poly(x, y) for (x, y) in nodes]).reshape(8, 1)
        exact_vals = np.array([poly(x, y) for (x, y) in sample_points])
        interpolated_vals = np.dot(N_samples.reshape(len(sample_points), 8), nodal_vals).flatten()
        max_error = np.max(np.abs(interpolated_vals - exact_vals))
        assert max_error < 1e-12

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    sample_points = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0.0, -0.7], [0.7, 0.0], [0.0, 0.7], [-0.7, 0.0]])
    (_, dN_nodes) = fcn(nodes)
    (_, dN_samples) = fcn(sample_points)
    polynomials = [(lambda x, y: 1.0, lambda x, y: (0.0, 0.0)), (lambda x, y: x, lambda x, y: (1.0, 0.0)), (lambda x, y: y, lambda x, y: (0.0, 1.0)), (lambda x, y: x * y, lambda x, y: (y, x)), (lambda x, y: x ** 2, lambda x, y: (2 * x, 0.0)), (lambda x, y: y ** 2, lambda x, y: (0.0, 2 * y))]
    for (poly, grad_poly) in polynomials:
        nodal_vals = np.array([poly(x, y) for (x, y) in nodes]).reshape(8, 1)
        exact_grads = np.array([grad_poly(x, y) for (x, y) in sample_points])
        interpolated_grads = np.zeros_like(exact_grads)
        for i in range(len(sample_points)):
            interpolated_grads[i] = np.dot(dN_samples[i].T, nodal_vals).flatten()
        max_error = np.max(np.abs(interpolated_grads - exact_grads))
        assert max_error < 1e-12