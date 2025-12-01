def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """
    The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised.
    """
    with pytest.raises(ValueError):
        fcn([0.0, 0.0])
    with pytest.raises(ValueError):
        fcn(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        fcn(np.zeros((5, 3)))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, np.inf], [0.0, 0.0]]))

def test_partition_of_unity_quad8(fcn):
    """
    Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance.
    """
    xi = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0.5, 0.5], [-0.3, 0.2]])
    (N, _) = fcn(xi)
    N_sum = np.sum(N, axis=1)
    np.testing.assert_allclose(N_sum, 1.0, atol=1e-14)

def test_derivative_partition_of_unity_quad8(fcn):
    """
    Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    xi = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0.2, -0.7]])
    (_, dN_dxi) = fcn(xi)
    dN_sum = np.sum(dN_dxi, axis=1)
    np.testing.assert_allclose(dN_sum, 0.0, atol=1e-14)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """
    For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity.
    """
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (N, _) = fcn(node_coords)
    M = N.squeeze(axis=-1)
    np.testing.assert_allclose(M, np.eye(8), atol=1e-14)

def test_value_completeness_quad8(fcn):
    """
    Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """

    def poly(p):
        (x, y) = (p[..., 0], p[..., 1])
        return 2.5 - 1.2 * x + 3.4 * y + 0.5 * x ** 2 - 2.0 * x * y + 1.1 * y ** 2
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    u_nodes = poly(nodes)
    test_points = np.array([[0.0, 0.0], [0.5, 0.5], [-0.2, 0.8], [0.9, -0.4]])
    (N, _) = fcn(test_points)
    u_approx = np.sum(N * u_nodes.reshape(1, 8, 1), axis=1).flatten()
    u_exact = poly(test_points)
    np.testing.assert_allclose(u_approx, u_exact, atol=1e-13)

def test_gradient_completeness_quad8(fcn):
    """
    Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """

    def poly(p):
        (x, y) = (p[..., 0], p[..., 1])
        return 2.5 - 1.2 * x + 3.4 * y + 0.5 * x ** 2 - 2.0 * x * y + 1.1 * y ** 2

    def grad_poly(p):
        (x, y) = (p[..., 0], p[..., 1])
        df_dx = -1.2 + 1.0 * x - 2.0 * y
        df_dy = 3.4 - 2.0 * x + 2.2 * y
        return np.stack([df_dx, df_dy], axis=-1)
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    u_nodes = poly(nodes)
    test_points = np.array([[0.0, 0.0], [0.5, -0.5], [-0.7, 0.1]])
    (_, dN_dxi) = fcn(test_points)
    grad_approx = np.sum(dN_dxi * u_nodes.reshape(1, 8, 1), axis=1)
    grad_exact = grad_poly(test_points)
    np.testing.assert_allclose(grad_approx, grad_exact, atol=1e-13)