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
        fcn(np.array([[0.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([-np.inf, 0.0]))

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    test_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0.5, 0.5], [-0.5, -0.5], [0.3, -0.7]])
    (N, _) = fcn(test_points)
    sum_N = np.sum(N, axis=1)
    assert np.allclose(sum_N, 1.0, atol=1e-12)

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    test_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0.5, 0.5], [-0.5, -0.5], [0.3, -0.7]])
    (_, dN_dxi) = fcn(test_points)
    sum_dN_dxi = np.sum(dN_dxi, axis=1)
    assert np.allclose(sum_dN_dxi, 0.0, atol=1e-12)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (N, _) = fcn(nodes)
    N_matrix = N.reshape(8, 8)
    expected_identity = np.eye(8)
    assert np.allclose(N_matrix, expected_identity, atol=1e-12)

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    test_points = np.array([[0.0, 0.0], [0.5, 0.3], [-0.7, 0.2], [0.8, -0.6]])
    (N_test, _) = fcn(test_points)
    (a, b, c) = (2.0, 3.0, 1.5)
    f_linear = lambda x, y: a + b * x + c * y
    nodal_values_linear = f_linear(nodes[:, 0], nodes[:, 1])
    interpolated_linear = np.sum(N_test * nodal_values_linear, axis=1)
    exact_linear = f_linear(test_points[:, 0], test_points[:, 1])
    assert np.max(np.abs(interpolated_linear - exact_linear)) < 1e-12
    (a, b, c, d, e, f) = (2.0, 3.0, 1.5, 0.5, 1.2, 0.8)
    f_quadratic = lambda x, y: a + b * x + c * y + d * x ** 2 + e * x * y + f * y ** 2
    nodal_values_quadratic = f_quadratic(nodes[:, 0], nodes[:, 1])
    interpolated_quadratic = np.sum(N_test * nodal_values_quadratic, axis=1)
    exact_quadratic = f_quadratic(test_points[:, 0], test_points[:, 1])
    assert np.max(np.abs(interpolated_quadratic - exact_quadratic)) < 1e-12

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    test_points = np.array([[0.0, 0.0], [0.5, 0.3], [-0.7, 0.2], [0.8, -0.6]])
    (N_test, dN_dxi_test) = fcn(test_points)
    (a, b, c) = (2.0, 3.0, 1.5)
    f_linear = lambda x, y: a + b * x + c * y
    grad_f_linear = lambda x, y: np.array