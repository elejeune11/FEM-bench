def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """
    D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised.
    """
    with pytest.raises(ValueError):
        fcn([0.1, 0.1])
    with pytest.raises(ValueError):
        fcn(np.array([0.1, 0.1, 0.1]))
    with pytest.raises(ValueError):
        fcn(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.5]))
    with pytest.raises(ValueError):
        fcn(np.array([0.5, np.inf]))

def test_partition_of_unity_tri6(fcn):
    """
    Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures 
    that the sum equals 1 within tight tolerance.
    """
    xi_test = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.33, 0.33], [0.2, 0.6]])
    (N, _) = fcn(xi_test)
    N_sum = np.sum(N, axis=1).flatten()
    np.testing.assert_allclose(N_sum, 1.0, atol=1e-14, rtol=0)

def test_derivative_partition_of_unity_tri6(fcn):
    """
    Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    xi_test = np.array([[0.1, 0.1], [0.7, 0.1], [0.1, 0.7], [0.333, 0.333]])
    (_, dN_dxi) = fcn(xi_test)
    grad_sum = np.sum(dN_dxi, axis=1)
    np.testing.assert_allclose(grad_sum, 0.0, atol=1e-14, rtol=0)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """
    For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N, _) = fcn(nodes)
    eval_matrix = N[:, :, 0]
    np.testing.assert_allclose(eval_matrix, np.eye(6), atol=1e-14, rtol=0)

def test_value_completeness_tri6(fcn):
    """
    Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    coeffs = [2.5, -1.0, 3.0, 0.5, 2.0, -1.5]

    def exact_field(pts):
        (x, y) = (pts[:, 0], pts[:, 1])
        return coeffs[0] + coeffs[1] * x + coeffs[2] * y + coeffs[3] * x ** 2 + coeffs[4] * y ** 2 + coeffs[5] * x * y
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    u_nodes = exact_field(nodes)
    xi_test = np.array([[0.2, 0.2], [0.8, 0.1], [0.1, 0.8], [0.4, 0.3]])
    (N, _) = fcn(xi_test)
    u_interp = np.sum(N[:, :, 0] * u_nodes, axis=1)
    u_exact = exact_field(xi_test)
    np.testing.assert_allclose(u_interp, u_exact, atol=1e-14, rtol=0)

def test_gradient_completeness_tri6(fcn):
    """
    Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    coeffs = [1.0, 2.0, -0.5, 1.5, -1.0, 0.8]

    def exact_field(pts):
        (x, y) = (pts[:, 0], pts[:, 1])
        return coeffs[0] + coeffs[1] * x + coeffs[2] * y + coeffs[3] * x ** 2 + coeffs[4] * y ** 2 + coeffs[5] * x * y

    def exact_gradient(pts):
        (x, y) = (pts[:, 0], pts[:, 1])
        dp_dx = coeffs[1] + 2 * coeffs[3] * x + coeffs[5] * y
        dp_dy = coeffs[2] + 2 * coeffs[4] * y + coeffs[5] * x
        return np.stack((dp_dx, dp_dy), axis=1)
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    u_nodes = exact_field(nodes)
    xi_test = np.array([[0.25, 0.25], [0.1, 0.5], [0.6, 0.2]])
    (_, dN_dxi) = fcn(xi_test)
    grad_interp = np.einsum('nij,i->nj', dN_dxi, u_nodes)
    grad_exact = exact_gradient(xi_test)
    np.testing.assert_allclose(grad_interp, grad_exact, atol=1e-13, rtol=0)