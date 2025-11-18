def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    with pytest.raises(ValueError):
        fcn([0.0, 0.0])
    with pytest.raises(ValueError):
        fcn((0.0, 0.0))
    with pytest.raises(ValueError):
        fcn(np.array([0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0], [0.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([[[0.0, 0.0]]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([0.0, np.inf]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, -np.inf], [0.0, 0.0]]))

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    test_points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
    (N, _) = fcn(test_points)
    sums = np.sum(N, axis=1).flatten()
    assert np.allclose(sums, 1.0, rtol=1e-12, atol=1e-12)

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    test_points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
    (_, dN_dxi) = fcn(test_points)
    grad_sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(grad_sums, 0.0, rtol=1e-12, atol=1e-12)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (N, _) = fcn(nodes)
    kronecker_matrix = N[:, :, 0].T
    assert np.allclose(kronecker_matrix, np.eye(8), rtol=1e-12, atol=1e-12)

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    sample_points = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5], [0.3, -0.7], [-0.3, 0.7], [0.8, 0.2], [-0.8, -0.2]])
    polynomials = [lambda xi, eta: 1.0, lambda xi, eta: xi, lambda xi, eta: eta, lambda xi, eta: xi + eta, lambda xi, eta: xi * eta, lambda xi, eta: xi ** 2, lambda xi, eta: eta ** 2, lambda xi, eta: xi ** 2 + eta ** 2, lambda xi, eta: xi ** 2 - eta ** 2, lambda xi, eta: 2 * xi ** 2 + 3 * xi * eta - eta ** 2]
    for poly in polynomials:
        nodal_values = np.array([poly(node[0], node[1]) for node in nodes])
        (N, _) = fcn(sample_points)
        interpolated = np.sum(N * nodal_values[np.newaxis, :, np.newaxis], axis=1).flatten()
        exact = np.array([poly(pt[0], pt[1]) for pt in sample_points])
        assert np.allclose(interpolated, exact, rtol=1e-12, atol=1e-12)

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    sample_points = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5], [0.3, -0.7], [-0.3, 0.7], [0.8, 0.2], [-0.8, -0.2]])
    test_cases = [(lambda xi, eta: 1.0, lambda xi, eta: [0.0, 0.0]), (lambda xi, eta: xi, lambda xi, eta: [1.0, 0.0]), (lambda xi, eta: eta, lambda xi, eta: [0.0, 1.0]), (lambda xi, eta: xi + eta, lambda xi, eta: [1.0, 1.0]), (lambda xi, eta: xi * eta, lambda xi, eta: [eta, xi]), (lambda xi, eta: xi ** 2, lambda xi, eta: [2 * xi, 0.0]), (lambda xi, eta: eta ** 2, lambda xi, eta: [0.0, 2 * eta]), (lambda xi, eta: xi ** 2 + eta ** 2, lambda xi, eta: [2 * xi, 2 * eta]), (lambda xi, eta: xi ** 2 - eta ** 2, lambda xi, eta: [2 * xi, -2 * eta]), (lambda xi, eta: 2 * xi ** 2 + 3 * xi * eta - eta ** 2, lambda xi, eta: [4 * xi + 3 * eta, 3 * xi - 2 * eta])]
    for (poly, grad_poly) in test_cases:
        nodal_values = np.array([poly(node[0], node[1]) for node in nodes])
        (_, dN_dxi) = fcn(sample_points)
        reconstructed_grad = np.sum(dN_dxi * nodal_values[np.newaxis, :, np.newaxis], axis=1)
        exact_grad = np.array([grad_poly(pt[0], pt[1]) for pt in sample_points])
        assert np.allclose(reconstructed_grad, exact_grad, rtol=1e-12, atol=1e-12)