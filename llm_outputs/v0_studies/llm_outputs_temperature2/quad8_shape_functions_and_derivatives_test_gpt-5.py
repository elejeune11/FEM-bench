def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """
    The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised.
    """
    bad_inputs = [None, 0, 'xi', [0.0, 0.0], (0.0, 0.0), np.array(0.0), np.array([0.0, 0.0, 0.0]), np.array([[0.0], [0.0]]), np.array([[0.0, 0.0, 0.0]]), np.array([np.nan, 0.0]), np.array([[0.0, np.inf]]), np.array([[0.0, 0.0], [np.nan, 0.0]])]
    for bad in bad_inputs:
        with pytest.raises(ValueError):
            fcn(bad)

def test_partition_of_unity_quad8(fcn):
    """
    Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance.
    """
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    (N, dN) = fcn(pts)
    sum_N = N[:, :, 0].sum(axis=1)
    assert np.allclose(sum_N, np.ones(pts.shape[0]), rtol=0.0, atol=1e-14)

def test_derivative_partition_of_unity_quad8(fcn):
    """
    Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    (_, dN) = fcn(pts)
    grad_sum = dN.sum(axis=1)
    assert np.allclose(grad_sum, np.zeros((pts.shape[0], 2)), rtol=0.0, atol=1e-14)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """
    For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (N, _) = fcn(nodes)
    M = N[:, :, 0].T
    assert np.allclose(M, np.eye(8), rtol=0.0, atol=1e-14)

def test_value_completeness_quad8(fcn):
    """
    Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    xi_vals = np.array([-1.0, -0.5, 0.0, 0.4, 0.9, 1.0])
    eta_vals = np.array([-1.0, -0.7, -0.1, 0.5, 0.8, 1.0])
    pts = np.array([[x, y] for x in xi_vals for y in eta_vals], dtype=float)
    (a0, a1, a2) = (0.3, -1.1, 0.7)

    def p_lin(x, y):
        return a0 + a1 * x + a2 * y
    (b0, b1, b2, b3, b4, b5) = (0.2, -0.8, 0.6, 1.1, -0.33, 0.9)

    def p_quad(x, y):
        return b0 + b1 * x + b2 * y + b3 * x * y + b4 * x * x + b5 * y * y
    u_nodes_lin = np.array([p_lin(x, y) for (x, y) in nodes])
    u_nodes_quad = np.array([p_quad(x, y) for (x, y) in nodes])
    (N_pts, _) = fcn(pts)
    Nm = N_pts[:, :, 0]
    u_lin_pred = Nm @ u_nodes_lin
    u_quad_pred = Nm @ u_nodes_quad
    u_lin_exact = np.array([p_lin(x, y) for (x, y) in pts])
    u_quad_exact = np.array([p_quad(x, y) for (x, y) in pts])
    err_lin = np.max(np.abs(u_lin_pred - u_lin_exact))
    err_quad = np.max(np.abs(u_quad_pred - u_quad_exact))
    assert err_lin < 1e-12
    assert err_quad < 1e-12

def test_gradient_completeness_quad8(fcn):
    """
    Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    xi_vals = np.array([-1.0, -0.6, -0.2, 0.3, 0.75, 1.0])
    eta_vals = np.array([-1.0, -0.8, -0.1, 0.4, 0.95, 1.0])
    pts = np.array([[x, y] for x in xi_vals for y in eta_vals], dtype=float)
    (a0, a1, a2) = (-0.4, 0.9, -1.3)

    def p_lin(x, y):
        return a0 + a1 * x + a2 * y

    def grad_lin(x, y):
        return np.array([a1, a2])
    (b0, b1, b2, b3, b4, b5) = (0.1, -0.7, 0.55, -1.2, 0.45, -0.8)

    def p_quad(x, y):
        return b0 + b1 * x + b2 * y + b3 * x * y + b4 * x * x + b5 * y * y

    def grad_quad(x, y):
        gx = b1 + b3 * y + 2.0 * b4 * x
        gy = b2 + b3 * x + 2.0 * b5 * y
        return np.array([gx, gy])
    u_nodes_lin = np.array([p_lin(x, y) for (x, y) in nodes])
    u_nodes_quad = np.array([p_quad(x, y) for (x, y) in nodes])
    (_, dN_pts) = fcn(pts)
    grad_lin_pred = (dN_pts * u_nodes_lin[np.newaxis, :, np.newaxis]).sum(axis=1)
    grad_quad_pred = (dN_pts * u_nodes_quad[np.newaxis, :, np.newaxis]).sum(axis=1)
    grad_lin_exact = np.array([grad_lin(x, y) for (x, y) in pts])
    grad_quad_exact = np.array([grad_quad(x, y) for (x, y) in pts])
    err_lin = np.max(np.linalg.norm(grad_lin_pred - grad_lin_exact, axis=1))
    err_quad = np.max(np.linalg.norm(grad_quad_pred - grad_quad_exact, axis=1))
    assert err_lin < 1e-12
    assert err_quad < 1e-12