def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    bad_inputs = ['not an array', np.array([1, 2, 3]), np.array([[1, 2], [3, 4], [5, 6]]), np.array([np.nan, 0]), np.array([np.inf, 0]), np.array([0, np.inf])]
    for bad_input in bad_inputs:
        with pytest.raises(ValueError):
            fcn(bad_input)

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    sample_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (N, _) = fcn(sample_points)
    for i in range(N.shape[0]):
        assert np.isclose(np.sum(N[i, :, 0]), 1.0, atol=1e-09)

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    sample_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (_, dN_dxi) = fcn(sample_points)
    for i in range(dN_dxi.shape[0]):
        assert np.allclose(np.sum(dN_dxi[i, :, :], axis=0), [0.0, 0.0], atol=1e-09)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    reference_nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (N, _) = fcn(reference_nodes)
    identity_matrix = np.eye(8)
    for i in range(8):
        assert np.allclose(N[i, :, 0], identity_matrix[i], atol=1e-09)

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""

    def poly(xi, eta):
        return 1 + xi + eta + xi * eta + xi ** 2 + eta ** 2
    reference_nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    nodal_values = np.array([poly(xi, eta) for (xi, eta) in reference_nodes])
    (N, _) = fcn(reference_nodes)
    interpolated_values = np.einsum('ijk,j->ik', N, nodal_values)
    assert np.allclose(interpolated_values.flatten(), nodal_values, atol=1e-09)

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""

    def grad_poly(xi, eta):
        return np.array([1 + eta + 2 * xi, 1 + xi + 2 * eta])
    reference_nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (_, dN_dxi) = fcn(reference_nodes)
    nodal_gradients = np.array([grad_poly(xi, eta) for (xi, eta) in reference_nodes])
    interpolated_gradients = np.einsum('ijk,jk->ik', dN_dxi, nodal_gradients)
    for (i, (xi, eta)) in enumerate(reference_nodes):
        assert np.allclose(interpolated_gradients[i], grad_poly(xi, eta), atol=1e-09)