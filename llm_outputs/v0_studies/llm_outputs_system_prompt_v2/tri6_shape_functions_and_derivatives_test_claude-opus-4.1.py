def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised."""
    with pytest.raises(ValueError):
        fcn([0.5, 0.5])
    with pytest.raises(ValueError):
        fcn(np.array([0.5]))
    with pytest.raises(ValueError):
        fcn(np.array([[[0.5, 0.5]]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.5, 0.5, 0.5]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.5]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.5, np.inf]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.5, 0.5], [-np.inf, 0.2]]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures 
    that the sum equals 1 within tight tolerance."""
    test_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [1 / 3, 1 / 3], [0.2, 0.3], [0.1, 0.7], [0.6, 0.2]])
    (N, _) = fcn(test_points)
    sums = np.sum(N, axis=1).flatten()
    assert np.allclose(sums, 1.0, rtol=1e-14, atol=1e-14)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    test_points = np.array([[1 / 3, 1 / 3], [0.2, 0.3], [0.1, 0.7], [0.6, 0.2], [0.25, 0.25], [0.4, 0.4], [0.15, 0.45]])
    (_, dN_dxi) = fcn(test_points)
    grad_sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(grad_sums, 0.0, rtol=1e-14, atol=1e-14)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N, _) = fcn(nodes)
    kronecker_matrix = N[:, :, 0].T
    assert np.allclose(kronecker_matrix, np.eye(6), rtol=1e-14, atol=1e-14)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    test_points = np.array([[0.2, 0.3], [0.1, 0.7], [0.6, 0.2], [0.25, 0.25], [0.4, 0.4]])
    polynomials = [lambda xi, eta: 1.0, lambda xi, eta: xi, lambda xi, eta: eta, lambda xi, eta: xi ** 2, lambda xi, eta: xi * eta, lambda xi, eta: eta ** 2]
    for poly in polynomials:
        nodal_values = np.array([poly(node[0], node[1]) for node in nodes])
        (N_test, _) = fcn(test_points)
        interpolated = np.sum(N_test * nodal_values[np.newaxis, :, np.newaxis], axis=1).flatten()
        exact = np.array([poly(pt[0], pt[1]) for pt in test_points])
        assert np.allclose(interpolated, exact, rtol=1e-14, atol=1e-14)

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    test_points = np.array([[0.2, 0.3], [0.1, 0.7], [0.6, 0.2], [0.25, 0.25], [0.4, 0.4]])
    poly_grad_pairs = [(lambda xi, eta: 1.0, lambda xi, eta: np.array([0.0, 0.0])), (lambda xi, eta: xi, lambda xi, eta: np.array([1.0, 0.0])), (lambda xi, eta: eta, lambda xi, eta: np.array([0.0, 1.0])), (lambda xi, eta: xi ** 2, lambda xi, eta: np.array([2 * xi, 0.0])), (lambda xi, eta: xi * eta, lambda xi, eta: np.array([eta, xi])), (lambda xi, eta: eta ** 2, lambda xi, eta: np.array([0.0, 2 * eta]))]
    for (poly, grad_func) in poly_grad_pairs:
        nodal_values = np.array([poly(node[0], node[1]) for node in nodes])
        (_, dN_dxi_test) = fcn(test_points)
        interpolated_grad = np.sum(dN_dxi_test * nodal_values[np.newaxis, :, np.newaxis], axis=1)
        exact_grad = np.array([grad_func(pt[0], pt[1]) for pt in test_points])
        assert np.allclose(interpolated_grad, exact_grad, rtol=1e-14, atol=1e-14)