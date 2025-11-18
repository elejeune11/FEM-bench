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
        fcn(np.array([[[0.0, 0.0]]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, -np.inf], [0.0, 0.0]]))

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    sample_points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
    (N, _) = fcn(sample_points)
    for i in range(sample_points.shape[0]):
        sum_N = np.sum(N[i, :, 0])
        assert np.abs(sum_N - 1.0) < 1e-12

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    sample_points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
    (_, dN_dxi) = fcn(sample_points)
    for i in range(sample_points.shape[0]):
        sum_dN_dxi = np.sum(dN_dxi[i, :, 0])
        sum_dN_deta = np.sum(dN_dxi[i, :, 1])
        assert np.abs(sum_dN_dxi) < 1e-12
        assert np.abs(sum_dN_deta) < 1e-12

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (N, _) = fcn(nodes)
    kronecker_matrix = N[:, :, 0].T
    identity = np.eye(8)
    assert np.allclose(kronecker_matrix, identity, atol=1e-12)

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    sample_points = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5], [0.3, -0.7], [-0.3, 0.7]])
    nodal_values = np.ones(8)
    (N, _) = fcn(sample_points)
    interpolated = N @ nodal_values.reshape(-1, 1)
    exact = np.ones((sample_points.shape[0], 1))
    assert np.max(np.abs(interpolated - exact)) < 1e-12
    nodal_values = 2 * nodes[:, 0] + 3 * nodes[:, 1] + 1
    (N, _) = fcn(sample_points)
    interpolated = N @ nodal_values.reshape(-1, 1)
    exact = (2 * sample_points[:, 0] + 3 * sample_points[:, 1] + 1).reshape(-1, 1)
    assert np.max(np.abs(interpolated - exact)) < 1e-12
    nodal_values = nodes[:, 0] ** 2 + 2 * nodes[:, 0] * nodes[:, 1] + nodes[:, 1] ** 2
    (N, _) = fcn(sample_points)
    interpolated = N @ nodal_values.reshape(-1, 1)
    exact = (sample_points[:, 0] ** 2 + 2 * sample_points[:, 0] * sample_points[:, 1] + sample_points[:, 1] ** 2).reshape(-1, 1)
    assert np.max(np.abs(interpolated - exact)) < 1e-12

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    sample_points = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5], [0.3, -0.7], [-0.3, 0.7]])
    nodal_values = np.ones(8)
    (_, dN_dxi) = fcn(sample_points)
    grad_interpolated = dN_dxi @ nodal_values.reshape(-1, 1)
    exact_grad = np.zeros((sample_points.shape[0], 2))
    assert np.max(np.abs(grad_interpolated.squeeze() - exact_grad)) < 1e-12
    nodal_values = 2 * nodes[:, 0] + 3 * nodes[:, 1] + 1
    (_, dN_dxi) = fcn(sample_points)
    grad_interpolated = dN_dxi @ nodal_values.reshape(-1, 1)
    exact_grad = np.tile([2.0, 3.0], (sample_points.shape[0], 1))
    assert np.max(np.abs(grad_interpolated.squeeze() - exact_grad)) < 1e-12
    nodal_values = nodes[:, 0] ** 2 + 2 * nodes[:, 0] * nodes[:, 1] + nodes[:, 1] ** 2
    (_, dN_dxi) = fcn(sample_points)
    grad_interpolated = dN_dxi @ nodal_values.reshape(-1, 1)
    exact_grad = np.column_stack([2 * sample_points[:, 0] + 2 * sample_points[:, 1], 2 * sample_points[:, 0] + 2 * sample_points[:, 1]])
    assert np.max(np.abs(grad_interpolated.squeeze() - exact_grad)) < 1e-12