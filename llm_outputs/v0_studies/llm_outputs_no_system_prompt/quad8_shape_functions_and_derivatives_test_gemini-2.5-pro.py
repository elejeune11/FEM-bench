def test_quad8_shape_functions_and_derivatives_input_errors(fcn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
with finite values. Invalid inputs must raise ValueError. This test feeds a set of
bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    bad_inputs = [[0, 0], (0, 0), 'not an array', None, np.array(1), np.array([1, 2, 3]), np.array([[1, 2, 3], [4, 5, 6]]), np.array([[[0, 0]]]), np.array([np.nan, 0]), np.array([0, np.inf]), np.array([-np.inf, 0]), np.array([[0, 0], [np.nan, 1]])]
    for bad_input in bad_inputs:
        with pytest.raises(ValueError):
            fcn(bad_input)

def test_partition_of_unity_quad8(fcn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
and centroid) and ensures that the sum equals 1 within tight tolerance."""
    xi_points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.25, 0.75], [1.0 / 3.0, -1.0 / 3.0]])
    (N, _) = fcn(xi_points)
    sum_N = np.sum(N, axis=1)
    assert_allclose(sum_N, 1.0, atol=1e-15)

def test_derivative_partition_of_unity_quad8(fcn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
For every sample point, the vector sum equals (0,0) within tight tolerance."""
    xi_points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.25, 0.75], [1.0 / 3.0, -1.0 / 3.0]])
    (_, dN_dxi) = fcn(xi_points)
    sum_dN = np.sum(dN_dxi, axis=1)
    expected = np.zeros_like(sum_dN)
    assert_allclose(sum_dN, expected, atol=1e-15)

def test_kronecker_delta_at_nodes_quad8(fcn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (N, _) = fcn(node_coords)
    N_at_nodes = N.squeeze(axis=2).T
    expected = np.eye(8)
    assert_allclose(N_at_nodes, expected, atol=1e-15)

def test_value_completeness_quad8(fcn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
degree-1 and degree-2 polynomials. Nodal values are set from the exact
polynomial, the field is interpolated at sample points, and the maximum
error is verified to be nearly zero."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    xi_points = np.array([[0.0, 0.0], [0.5, 0.5], [-0.25, 0.75], [1.0 / 3.0, -1.0 / 3.0]])

    def poly(xi, eta):
        return 2.0 - 3.0 * xi + 4.0 * eta + 5.0 * xi * eta - 6.0 * xi ** 2 + 7.0 * eta ** 2
    p_nodes = poly(node_coords[:, 0], node_coords[:, 1])
    (N, _) = fcn(xi_points)
    p_interp = np.einsum('ijk,j->ik', N, p_nodes).squeeze(axis=1)
    p_exact = poly(xi_points[:, 0], xi_points[:, 1])
    assert_allclose(p_interp, p_exact, atol=1e-14)

def test_gradient_completeness_quad8(fcn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]):
    """Check that Q8 quad shape functions reproduce the exact gradient for
degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
values and compared with the analytic gradient at sample points, with
maximum error verified to be nearly zero."""
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    xi_points = np.array([[0.0, 0.0], [0.5, 0.5], [-0.25, 0.75], [1.0 / 3.0, -1.0 / 3.0]])

    def poly(xi, eta):
        return 2.0 - 3.0 * xi + 4.0 * eta + 5.0 * xi * eta - 6.0 * xi ** 2 + 7.0 * eta ** 2

    def grad_poly(xi, eta):
        dp_dxi = -3.0 + 5.0 * eta - 12.0 * xi
        dp_deta = 4.0 + 5.0 * xi + 14.0 * eta
        return np.stack([dp_dxi, dp_deta], axis=-1)
    p_nodes = poly(node_coords[:, 0], node_coords[:, 1])
    (_, dN_dxi) = fcn(xi_points)
    grad_p_interp = np.einsum('j,ijk->ik', p_nodes, dN_dxi)
    grad_p_exact = grad_poly(xi_points[:, 0], xi_points[:, 1])
    assert_allclose(grad_p_interp, grad_p_exact, atol=1e-14)