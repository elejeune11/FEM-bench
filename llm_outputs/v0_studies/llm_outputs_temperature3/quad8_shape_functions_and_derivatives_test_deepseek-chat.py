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
        fcn(np.array([np.inf, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, np.nan]]))

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    test_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (N, _) = fcn(test_points)
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-12)

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    test_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (_, dN_dxi) = fcn(test_points)
    gradient_sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(gradient_sums, 0.0, atol=1e-12)

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
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    test_points = np.array([[0.0, 0.0], [0.5, -0.5], [-0.5, 0.5], [0.3, 0.7]])
    linear_polys = [lambda xi, eta: 1.0, lambda xi, eta: xi, lambda xi, eta: eta, lambda xi, eta: 0.5 * xi + 0.3 * eta + 1.0]
    quadratic_polys = [lambda xi, eta: xi * eta, lambda xi, eta: xi ** 2, lambda xi, eta: eta ** 2, lambda xi, eta: 0.5 * xi ** 2 + 0.3 * eta ** 2 + 0.2 * xi * eta + xi - 0.5 * eta + 2.0]
    all_polys = linear_polys + quadratic_polys
    for poly in all_polys:
        nodal_values = np.array([poly(node[0], node[1]) for node in nodes])
        (N_test, _) = fcn(test_points)
        N_test = N_test.reshape(len(test_points), 8)
        interpolated = N_test @ nodal_values
        exact = np.array([poly(pt[0], pt[1]) for pt in test_points])
        max_error = np.max(np.abs(interpolated - exact))
        assert max_error < 1e-12

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    test_points = np.array([[0.0, 0.0], [0.5, -0.5], [-0.5, 0.5], [0.3, 0.7]])
    linear_cases = [(lambda xi, eta: 1.0, lambda xi, eta: [0.0, 0.0]), (lambda xi, eta: xi, lambda xi, eta: [1.0, 0.0]), (lambda xi, eta: eta, lambda xi, eta: [0.0, 1.0]), (lambda xi, eta: 0.5 * xi + 0.3 * eta + 1.0, lambda xi, eta: [0.5, 0.3])]
    quadratic_cases = [(lambda xi, eta: xi * eta, lambda xi, eta: [eta, xi]), (lambda xi, eta: xi ** 2, lambda xi, eta: [2 * xi, 0.0]), (lambda xi, eta: eta ** 2, lambda xi, eta: [0.0, 2 * eta]), (lambda xi, eta: 0.5 * xi ** 2 + 0.3 * eta ** 2 + 0.2 * xi * eta + xi - 0.5 * eta + 2.0, lambda xi, eta: [xi + 0.2 * eta + 1.0, 0.6 * eta + 0.2 * xi - 0.5])]
    all_cases = linear_cases + quadratic_cases
    for (poly, grad_poly) in all_cases:
        nodal_values = np.array([poly(node[0], node[1]) for node in nodes])
        (N_test, dN_dxi_test) = fcn(test_points)
        N_test = N_test.reshape(len(test_points), 8)
        dN_dxi_test = dN_dxi_test.reshape(len(test_points), 8, 2)
        reconstructed_grad = np.zeros((len(test_points), 2))
        for i in range(len(test_points)):
            for j in range(8):
                reconstructed_grad[i] += nodal_values[j] * dN_dxi_test[i, j]
        exact_grad = np.array([grad_poly(pt[0], pt[1]) for pt in test_points])
        max_error = np.max(np.abs(reconstructed_grad - exact_grad))
        assert max_error < 1e-12