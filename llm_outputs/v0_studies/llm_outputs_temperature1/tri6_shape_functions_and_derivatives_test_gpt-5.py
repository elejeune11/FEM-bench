def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """
    D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised.
    """
    with pytest.raises(ValueError):
        fcn([0.2, 0.3])
    with pytest.raises(ValueError):
        fcn(np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.1], [0.2]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.2]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.1, np.inf], [0.2, 0.3]]))

def test_partition_of_unity_tri6(fcn):
    """
    Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures 
    that the sum equals 1 within tight tolerance.
    """
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1 / 3, 1 / 3], [0.2, 0.3], [0.6, 0.2]], dtype=float)
    (N, dN) = fcn(pts)
    s = np.sum(N, axis=1).reshape(-1)
    assert np.allclose(s, np.ones_like(s), rtol=0, atol=1e-12)

def test_derivative_partition_of_unity_tri6(fcn):
    """
    Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.2, 0.3], [1 / 3, 1 / 3]], dtype=float)
    (N, dN) = fcn(pts)
    grad_sum = np.sum(dN, axis=1)
    assert np.allclose(grad_sum, np.zeros_like(grad_sum), rtol=0, atol=1e-12)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """
    For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    (N, _) = fcn(nodes)
    M = N[:, :, 0]
    assert np.allclose(M, np.eye(6), rtol=0, atol=1e-12)

def test_value_completeness_tri6(fcn):
    """
    Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.2, 0.3], [0.1, 0.2], [0.3, 0.4], [1 / 3, 1 / 3]], dtype=float)

    def eval_poly(c, x, y):
        return c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * y * y + c[5] * x * y
    coeffs_list = [np.array([2.0, -1.5, 0.75, 0.0, 0.0, 0.0]), np.array([0.3, 1.2, -0.7, 0.8, -0.55, 0.45]), np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])]
    (N_pts, _) = fcn(pts)
    N_pts = N_pts[:, :, 0]
    x_nodes = nodes[:, 0]
    y_nodes = nodes[:, 1]
    for c in coeffs_list:
        nodal_vals = eval_poly(c, x_nodes, y_nodes)
        interp_vals = N_pts @ nodal_vals
        exact_vals = eval_poly(c, pts[:, 0], pts[:, 1])
        err = np.max(np.abs(interp_vals - exact_vals))
        assert err <= 1e-12

def test_gradient_completeness_tri6(fcn):
    """
    Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.2, 0.3], [0.1, 0.2], [0.3, 0.4], [1 / 3, 1 / 3]], dtype=float)

    def grad_poly(c, x, y):
        gx = c[1] + 2.0 * c[3] * x + c[5] * y
        gy = c[2] + 2.0 * c[4] * y + c[5] * x
        return np.stack([gx, gy], axis=-1)
    coeffs_list = [np.array([2.0, -1.5, 0.75, 0.0, 0.0, 0.0]), np.array([0.3, 1.2, -0.7, 0.8, -0.55, 0.45])]
    x_nodes = nodes[:, 0]
    y_nodes = nodes[:, 1]
    for c in coeffs_list:
        nodal_vals = c[0] + c[1] * x_nodes + c[2] * y_nodes + c[3] * x_nodes ** 2 + c[4] * y_nodes ** 2 + c[5] * x_nodes * y_nodes
        (_, dN) = fcn(pts)
        recon_grad = np.sum(dN * nodal_vals[None, :, None], axis=1)
        exact_grad = grad_poly(c, pts[:, 0], pts[:, 1])
        err = np.max(np.abs(recon_grad - exact_grad))
        assert err <= 1e-12