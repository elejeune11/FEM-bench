def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised."""
    with pytest.raises(ValueError):
        fcn([0.1, 0.2])
    with pytest.raises(ValueError):
        fcn(np.array([0.1]))
    with pytest.raises(ValueError):
        fcn(np.array([[[0.1, 0.2]]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.1, 0.2, 0.3]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.2]))
    with pytest.raises(ValueError):
        fcn(np.array([0.1, np.inf]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures 
    that the sum equals 1 within tight tolerance."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1 / 3, 1 / 3], [0.1, 0.1], [0.2, 0.7], [0.6, 0.3]])
    (N, _) = fcn(sample_points)
    unity_sums = np.sum(N[:, :, 0], axis=1)
    assert np.allclose(unity_sums, 1.0, atol=1e-14)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    sample_points = np.array([[0.1, 0.1], [0.5, 0.0], [0.0, 0.5], [1 / 3, 1 / 3], [0.2, 0.7], [0.6, 0.3], [0.8, 0.1]])
    (_, dN_dxi) = fcn(sample_points)
    gradient_sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(gradient_sums, 0.0, atol=1e-14)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N, _) = fcn(nodes)
    shape_matrix = N[:, :, 0]
    identity = np.eye(6)
    assert np.allclose(shape_matrix, identity, atol=1e-14)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    test_points = np.array([[0.1, 0.1], [1 / 3, 1 / 3], [0.2, 0.6], [0.7, 0.2]])
    polynomials = [lambda xi, eta: 1.0, lambda xi, eta: xi, lambda xi, eta: eta, lambda xi, eta: xi + 2 * eta, lambda xi, eta: xi ** 2, lambda xi, eta: eta ** 2, lambda xi, eta: xi * eta]
    (N, _) = fcn(test_points)
    for poly in polynomials:
        nodal_values = np.array([poly(node[0], node[1]) for node in nodes])
        interpolated = np.dot(N[:, :, 0], nodal_values)
        exact = np.array([poly(pt[0], pt[1]) for pt in test_points])
        max_error = np.max(np.abs(interpolated - exact))
        assert max_error < 1e-14

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    test_points = np.array([[0.1, 0.1], [1 / 3, 1 / 3], [0.2, 0.6], [0.7, 0.2]])
    test_cases = [(lambda xi, eta: xi, lambda xi, eta: (1.0, 0.0)), (lambda xi, eta: eta, lambda xi, eta: (0.0, 1.0)), (lambda xi, eta: xi + 2 * eta, lambda xi, eta: (1.0, 2.0)), (lambda xi, eta: xi ** 2, lambda xi, eta: (2 * xi, 0.0)), (lambda xi, eta: eta ** 2, lambda xi, eta: (0.0, 2 * eta)), (lambda xi, eta: xi * eta, lambda xi, eta: (eta, xi))]
    (_, dN_dxi) = fcn(test_points)
    for (poly, grad_poly) in test_cases:
        nodal_values = np.array([poly(node[0], node[1]) for node in nodes])
        reconstructed_grad = np.zeros((len(test_points), 2))
        for i in range(len(test_points)):
            reconstructed_grad[i, 0] = np.dot(dN_dxi[i, :, 0], nodal_values)
            reconstructed_grad[i, 1] = np.dot(dN_dxi[i, :, 1], nodal_values)
        exact_grad = np.array([grad_poly(pt[0], pt[1]) for pt in test_points])
        max_error = np.max(np.abs(reconstructed_grad - exact_grad))
        assert max_error < 1e-14