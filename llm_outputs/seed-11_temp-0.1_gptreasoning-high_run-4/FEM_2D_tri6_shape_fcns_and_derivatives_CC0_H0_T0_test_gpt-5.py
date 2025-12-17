def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """
    D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised.
    """
    invalid_inputs = [[0.1, 0.2], np.array([0.1, 0.2, 0.3]), np.ones((1, 2, 1)), np.ones((2, 1)), np.array([np.nan, 0.2]), np.array([[0.1, np.inf], [0.2, 0.3]])]
    for xi in invalid_inputs:
        with pytest.raises(ValueError):
            fcn(xi)

def test_partition_of_unity_tri6(fcn):
    """
    Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures 
    that the sum equals 1 within tight tolerance.
    """
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.2, 0.3], [0.6, 0.3], [0.25, 0.25]], dtype=float)
    N, _ = fcn(pts)
    Nflat = N[..., 0] if N.ndim == 3 else N
    s = Nflat.sum(axis=1)
    assert np.allclose(s, np.ones_like(s), atol=1e-13, rtol=0.0)

def test_derivative_partition_of_unity_tri6(fcn):
    """
    Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.2, 0.3], [0.6, 0.1]], dtype=float)
    _, dN = fcn(pts)
    grad_sum = dN.sum(axis=1)
    assert np.allclose(grad_sum, 0.0, atol=1e-13, rtol=0.0)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """
    For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    N, _ = fcn(nodes)
    Nflat = N[..., 0] if N.ndim == 3 else N
    assert Nflat.shape == (6, 6)
    I = np.eye(6)
    assert np.allclose(Nflat, I, atol=1e-14, rtol=0.0)

def test_value_completeness_tri6(fcn):
    """
    Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.2, 0.3], [0.6, 0.3], [0.25, 0.25]], dtype=float)
    N, _ = fcn(pts)
    Nflat = N[..., 0] if N.ndim == 3 else N
    a0, a1, a2 = (0.5, -1.0 / 3.0, 2.0 / 3.0)

    def f_lin(xy):
        xi, eta = (xy[..., 0], xy[..., 1])
        return a0 + a1 * xi + a2 * eta
    f_nodes_lin = f_lin(nodes)
    f_interp_lin = Nflat @ f_nodes_lin
    f_true_lin = f_lin(pts)
    err_lin = np.max(np.abs(f_interp_lin - f_true_lin))
    assert err_lin < 1e-13
    b0, b1, b2, b3, b4, b5 = (0.123, -0.8, 0.9, 0.7, 0.4, -0.2)

    def f_quad(xy):
        xi, eta = (xy[..., 0], xy[..., 1])
        return b0 + b1 * xi + b2 * eta + b3 * xi * eta + b4 * xi ** 2 + b5 * eta ** 2
    f_nodes_quad = f_quad(nodes)
    f_interp_quad = Nflat @ f_nodes_quad
    f_true_quad = f_quad(pts)
    err_quad = np.max(np.abs(f_interp_quad - f_true_quad))
    assert err_quad < 1e-13

def test_gradient_completeness_tri6(fcn):
    """
    Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.2, 0.3], [0.6, 0.1], [0.25, 0.25]], dtype=float)
    _, dN = fcn(pts)
    a0, a1, a2 = (0.5, -1.0 / 3.0, 2.0 / 3.0)

    def f_lin_nodes(xy):
        xi, eta = (xy[..., 0], xy[..., 1])
        return a0 + a1 * xi + a2 * eta
    f_nodes_lin = f_lin_nodes(nodes)
    grad_interp_lin = np.tensordot(dN, f_nodes_lin, axes=([1], [0]))
    grad_true_lin = np.stack([np.full(pts.shape[0], a1), np.full(pts.shape[0], a2)], axis=1)
    err_lin = np.max(np.abs(grad_interp_lin - grad_true_lin))
    assert err_lin < 1e-13
    b0, b1, b2, b3, b4, b5 = (0.123, -0.8, 0.9, 0.7, 0.4, -0.2)

    def f_quad_nodes(xy):
        xi, eta = (xy[..., 0], xy[..., 1])
        return b0 + b1 * xi + b2 * eta + b3 * xi * eta + b4 * xi ** 2 + b5 * eta ** 2
    f_nodes_quad = f_quad_nodes(nodes)
    grad_interp_quad = np.tensordot(dN, f_nodes_quad, axes=([1], [0]))
    xi, eta = (pts[:, 0], pts[:, 1])
    grad_true_quad = np.stack([b1 + b3 * eta + 2.0 * b4 * xi, b2 + b3 * xi + 2.0 * b5 * eta], axis=1)
    err_quad = np.max(np.abs(grad_interp_quad - grad_true_quad))
    assert err_quad < 1e-12