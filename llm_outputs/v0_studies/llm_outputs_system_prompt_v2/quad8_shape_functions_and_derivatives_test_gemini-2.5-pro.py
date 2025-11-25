def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
with finite values. Invalid inputs must raise ValueError. This test feeds a set of
bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    with pytest.raises(ValueError):
        fcn('not a numpy array')
    with pytest.raises(ValueError):
        fcn([0.0, 0.0])
    with pytest.raises(ValueError):
        fcn(np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        fcn(np.array([[1, 2, 3], [4, 5, 6]]))
    with pytest.raises(ValueError):
        fcn(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([0.0, np.inf]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, 1.0], [np.nan, -1.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, 1.0], [0.5, np.inf]]))

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
and centroid) and ensures that the sum equals 1 within tight tolerance."""
    xi_pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, -0.5], [-0.25, 0.75], [1.0 / 3.0, -1.0 / 3.0]])
    (N, _) = fcn(xi_pts)
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1.0)

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
For every sample point, the vector sum equals (0,0) within tight tolerance."""
    xi_pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, -0.5], [-0.25, 0.75], [1.0 / 3.0, -1.0 / 3.0]])
    (_, dN_dxi) = fcn(xi_pts)
    grad_sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(grad_sums, 0.0)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    xi_nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (N, _) = fcn(xi_nodes)
    N_matrix = N.squeeze()
    assert np.allclose(N_matrix, np.eye(8))

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
degree-1 and degree-2 polynomials. Nodal values are set from the exact
polynomial, the field is interpolated at sample points, and the maximum
error is verified to be nearly zero."""
    xi_nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    poly = lambda xi, eta: 2.0 - 3.0 * xi + 4.0 * eta + 5.0 * xi * eta - 6.0 * xi ** 2 + 7.0 * eta ** 2
    p_nodes = poly(xi_nodes[:, 0], xi_nodes[:, 1])
    rng = np.random.default_rng(12345)
    xi_sample = rng.uniform(-1.0, 1.0, size=(20, 2))
    (N, _) = fcn(xi_sample)
    p_interp = np.dot(N.squeeze(), p_nodes)
    p_exact = poly(xi_sample[:, 0], xi_sample[:, 1])
    assert np.allclose(p_interp, p_exact)

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
values and compared with the analytic gradient at sample points, with
maximum error verified to be nearly zero."""
    xi_nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    poly = lambda xi, eta: 2.0 - 3.0 * xi + 4.0 * eta + 5.0 * xi * eta - 6.0 * xi ** 2 + 7.0 * eta ** 2
    grad_poly = lambda xi, eta: np.vstack([-3.0 + 5.0 * eta - 12.0 * xi, 4.0 + 5.0 * xi + 14.0 * eta]).T
    p_nodes = poly(xi_nodes[:, 0], xi_nodes[:, 1])
    rng = np.random.default_rng(54321)
    xi_sample = rng.uniform(-1.0, 1.0, size=(20, 2))
    (_, dN_dxi) = fcn(xi_sample)
    grad_p_interp = np.einsum('nji,j->ni', dN_dxi, p_nodes)
    grad_p_exact = grad_poly(xi_sample[:, 0], xi_sample[:, 1])
    assert np.allclose(grad_p_interp, grad_p_exact)