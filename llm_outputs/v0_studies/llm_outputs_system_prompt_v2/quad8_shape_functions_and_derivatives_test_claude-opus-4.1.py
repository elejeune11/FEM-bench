def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    with pytest.raises(ValueError):
        fcn([0.0, 0.0])
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
    test_points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
    (N, _) = fcn(test_points)
    for i in range(test_points.shape[0]):
        sum_N = np.sum(N[i, :, 0])
        assert np.abs(sum_N - 1.0) < 1e-12

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    test_points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
    (_, dN_dxi) = fcn(test_points)
    for i in range(test_points.shape[0]):
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
    test_points = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [0.3, -0.7], [-0.3, 0.7]])
    polynomials = [lambda xi, eta: 1.0, lambda xi, eta: xi, lambda xi, eta: eta, lambda xi, eta: xi + eta, lambda xi, eta: xi * eta, lambda xi, eta: xi ** 2, lambda xi, eta: eta ** 2, lambda xi, eta: xi ** 2 + eta ** 2, lambda xi, eta: xi ** 2 * eta, lambda xi, eta: xi * eta ** 2]
    for poly in polynomials:
        nodal_values = np.array([poly(node[0], node[1]) for node in nodes])
        (N, _) = fcn(test_points)
        for i in range(test_points.shape[0]):
            interpolated = np.sum(N[i, :, 0] * nodal_values)
            exact = poly(test_points[i, 0], test_points[i, 1])
            assert np.abs(interpolated - exact) < 1e-12

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    test_points = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [0.3, -0.7], [-0.3, 0.7]])
    polynomials_and_gradients = [(lambda xi, eta: 1.0, lambda xi, eta: (0.0, 0.0)), (lambda xi, eta: xi, lambda xi, eta: (1.0, 0.0)), (lambda xi, eta: eta, lambda xi, eta: (0.0, 1.0)), (lambda xi, eta: xi + eta, lambda xi, eta: (1.0, 1.0)), (lambda xi, eta: xi * eta, lambda xi, eta: (eta, xi)), (lambda xi, eta: xi ** 2, lambda xi, eta: (2 * xi, 0.0)), (lambda xi, eta: eta ** 2, lambda xi, eta: (0.0, 2 * eta)), (lambda xi, eta: xi ** 2 + eta ** 2, lambda xi, eta: (2 * xi, 2 * eta)), (lambda xi, eta: xi ** 2 * eta, lambda xi, eta: (2 * xi * eta, xi ** 2)), (lambda xi, eta: xi * eta ** 2, lambda xi, eta: (eta ** 2, 2 * xi * eta))]
    for (poly, grad_poly) in polynomials_and_gradients:
        nodal_values = np.array([poly(node[0], node[1]) for node in nodes])
        (_, dN_dxi) = fcn(test_points)
        for i in range(test_points.shape[0]):
            grad_interpolated = np.sum(dN_dxi[i, :, :] * nodal_values[:, np.newaxis], axis=0)
            grad_exact = grad_poly(test_points[i, 0], test_points[i, 1])
            assert np.abs(grad_interpolated[0] - grad_exact[0]) < 1e-12
            assert np.abs(grad_interpolated[1] - grad_exact[1]) < 1e-12