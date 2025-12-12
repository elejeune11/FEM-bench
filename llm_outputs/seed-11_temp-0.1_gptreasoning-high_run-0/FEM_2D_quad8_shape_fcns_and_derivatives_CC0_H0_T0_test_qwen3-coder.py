def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    with pytest.raises(ValueError):
        fcn([0.5, 0.5])
    with pytest.raises(ValueError):
        fcn(np.array([0.5]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.5, 0.5, 0.5]]))
    with pytest.raises(ValueError):
        fcn(np.array([[[0.5, 0.5]]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.5]))
    with pytest.raises(ValueError):
        fcn(np.array([0.5, np.inf]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.5, 0.5], [0.5, -np.inf]]))

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    sample_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (N, _) = fcn(sample_points)
    sums = np.sum(N.squeeze(), axis=1)
    expected = np.ones_like(sums)
    np.testing.assert_allclose(sums, expected, atol=1e-12)

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    sample_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (_, dN_dxi) = fcn(sample_points)
    grad_sums = np.sum(dN_dxi, axis=1)
    expected = np.zeros_like(grad_sums)
    np.testing.assert_allclose(grad_sums, expected, atol=1e-12)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (N, _) = fcn(node_coords)
    N_matrix = N.squeeze()
    expected = np.eye(8)
    np.testing.assert_allclose(N_matrix, expected, atol=1e-12)

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""

    def poly1(x, y):
        return 1.0

    def poly2(x, y):
        return x

    def poly3(x, y):
        return y

    def poly4(x, y):
        return x ** 2

    def poly5(x, y):
        return x * y

    def poly6(x, y):
        return y ** 2
    polys = [poly1, poly2, poly3, poly4, poly5, poly6]
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    sample_points = np.array([[0.3, -0.4], [-0.7, 0.2], [0.5, 0.5], [-0.3, -0.8]])
    for poly in polys:
        nodal_values = poly(node_coords[:, 0], node_coords[:, 1])
        (N, _) = fcn(sample_points)
        interpolated = np.sum(N.squeeze() * nodal_values, axis=1)
        exact = poly(sample_points[:, 0], sample_points[:, 1])
        max_error = np.max(np.abs(interpolated - exact))
        assert max_error < 1e-12

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""

    def poly1(x, y):
        return 1.0

    def grad1(x, y):
        return np.array([0.0, 0.0])

    def poly2(x, y):
        return x

    def grad2(x, y):
        return np.array([1.0, 0.0])

    def poly3(x, y):
        return y

    def grad3(x, y):
        return np.array([0.0, 1.0])

    def poly4(x, y):
        return x ** 2

    def grad4(x, y):
        return np.array([2 * x, 0.0])

    def poly5(x, y):
        return x * y

    def grad5(x, y):
        return np.array([y, x])

    def poly6(x, y):
        return y ** 2

    def grad6(x, y):
        return np.array([0.0, 2 * y])
    poly_grad_pairs = [(poly1, grad1), (poly2, grad2), (poly3, grad3), (poly4, grad4), (poly5, grad5), (poly6, grad6)]
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    sample_points = np.array([[0.3, -0.4], [-0.7, 0.2], [0.5, 0.5], [-0.3, -0.8]])
    for (poly, grad) in poly_grad_pairs:
        nodal_values = poly(node_coords[:, 0], node_coords[:, 1])
        (_, dN_dxi) = fcn(sample_points)
        grad_reconstructed = np.sum(dN_dxi * nodal_values[:, np.newaxis], axis=1)
        exact_grads = np.array([grad(x, y) for (x, y) in sample_points])
        max_error = np.max(np.abs(grad_reconstructed - exact_grads))
        assert max_error < 1e-12