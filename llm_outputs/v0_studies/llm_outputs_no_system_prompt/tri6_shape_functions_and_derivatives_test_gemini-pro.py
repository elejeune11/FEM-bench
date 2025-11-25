def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised."""
    with pytest.raises(ValueError):
        fcn([1.0, 2.0])
    with pytest.raises(ValueError):
        fcn(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError):
        fcn(np.array([[np.inf, 2.0]]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures 
    that the sum equals 1 within tight tolerance."""
    xi = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (N, _) = fcn(xi)
    np.testing.assert_allclose(np.sum(N, axis=1), np.ones((xi.shape[0], 1)), atol=1e-15)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    xi = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (_, dN_dxi) = fcn(xi)
    np.testing.assert_allclose(np.sum(dN_dxi, axis=1), np.zeros((xi.shape[0], 2)), atol=1e-15)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    (N, _) = fcn(nodes)
    np.testing.assert_allclose(N[:, :, 0], np.eye(6), atol=1e-15)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    xi = np.random.rand(10, 2)
    x = 2 * xi - 1
    for (a, b, c) in [(1, 2, 3), (0, 1, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
        node_x = 2 * nodes - 1
        nodal_values = a + b * node_x[:, 0:1] + c * node_x[:, 1:2]
        (N, _) = fcn(xi)
        interpolated_values = np.sum(N * nodal_values, axis=1)
        exact_values = a + b * x[:, 0:1] + c * x[:, 1:2]
        np.testing.assert_allclose(interpolated_values, exact_values, atol=1e-14)

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    xi = np.random.rand(10, 2)
    for (a, b, c) in [(1, 2, 3), (0, 1, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
        node_x = 2 * nodes - 1
        nodal_values = a + b * node_x[:, 0:1] + c * node_x[:, 1:2]
        (N, dN_dxi) = fcn(xi)
        interpolated_gradient = np.einsum('nij,nk->ni', dN_dxi, nodal_values) * 2
        exact_gradient = np.tile(np.array([b, c]), (xi.shape[0], 1))
        np.testing.assert_allclose(interpolated_gradient, exact_gradient, atol=1e-14)