def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    with pytest.raises(ValueError):
        fcn([0.0, 0.0])
    with pytest.raises(ValueError):
        fcn(np.array([1.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([1.0, np.inf]))

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    test_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (N, _) = fcn(test_points)
    sum_N = np.sum(N, axis=1)
    assert np.allclose(sum_N, 1.0, atol=1e-12)

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    test_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (_, dN_dxi) = fcn(test_points)
    sum_dN = np.sum(dN_dxi, axis=1)
    assert np.allclose(sum_dN, 0.0, atol=1e-12)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (N, _) = fcn(nodes)
    N_matrix = N.reshape(8, 8)
    assert np.allclose(N_matrix, np.eye(8), atol=1e-12)

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    test_points = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0.0, -0.7], [0.7, 0.0], [0.0, 0.7], [-0.7, 0.0]])
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    nodal_values_linear = 2 * nodes[:, 0] + 3 * nodes[:, 1] + 1
    nodal_values_quadratic = nodes[:, 0] ** 2 + 2 * nodes[:, 0] * nodes[:, 1] + 3 * nodes[:, 1] ** 2 + 4 * nodes[:, 0] + 5 * nodes[:, 1] + 6
    (N, _) = fcn(test_points)
    N_reshaped = N.reshape(len(test_points), 8)
    interpolated_linear = N_reshaped @ nodal_values_linear
    exact_linear = 2 * test_points[:, 0] + 3 * test_points[:, 1] + 1
    linear_error = np.max(np.abs(interpolated_linear - exact_linear))
    interpolated_quadratic = N_reshaped @ nodal_values_quadratic
    exact_quadratic = test_points[:, 0] ** 2 + 2 * test_points[:, 0] * test_points[:, 1] + 3 * test_points[:, 1] ** 2 + 4 * test_points[:, 0] + 5 * test_points[:, 1] + 6
    quadratic_error = np.max(np.abs(interpolated_quadratic - exact_quadratic))
    assert linear_error < 1e-12
    assert quadratic_error < 1e-12

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    test_points = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0.0, -0.7], [0.7, 0.0], [0.0, 0.7], [-0.7, 0.0]])
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    nodal_values_linear = 2 * nodes[:, 0] + 3 * nodes[:, 1] + 1
    exact_grad_linear = np.array([2.0, 3.0])
    nodal_values_quadratic = nodes[:, 0] ** 2 + 2 * nodes[:, 0] * nodes[:, 1] + 3 * nodes[:, 1] ** 2 + 4 * nodes[:, 0] + 5 * nodes[:, 1] + 6
    (N, dN_dxi) = fcn(test_points)
    max_grad_error_linear = 0.0
    max_grad_error_quadratic = 0.0
    for (i, point) in enumerate(test_points):
        grad_reconstructed_linear = dN_dxi[i].T @ nodal_values_linear
        error_linear = np.max(np.abs(grad_reconstructed_linear - exact_grad_linear))
        max_grad_error_linear = max(max_grad_error_linear, error_linear)
        exact_grad_quadratic = np.array([2 * point[0] + 2 * point[1] + 4, 2 * point[0] + 6 * point[1] + 5])
        grad_reconstructed_quadratic = dN_dxi[i].T @ nodal_values_quadratic
        error_quadratic = np.max(np.abs(grad_reconstructed_quadratic - exact_grad_quadratic))
        max_grad_error_quadratic = max(max_grad_error_quadratic, error_quadratic)
    assert max_grad_error_linear < 1e-12
    assert max_grad_error_quadratic < 1e-12