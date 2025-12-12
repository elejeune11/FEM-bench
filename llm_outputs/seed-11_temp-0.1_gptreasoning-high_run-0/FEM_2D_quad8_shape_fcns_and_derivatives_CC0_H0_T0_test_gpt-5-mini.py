def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    bad_inputs = [None, [0.0, 0.0], (0.0, 0.0), np.array([0.0, 1.0, 2.0]), np.array([[0.0], [1.0]]), np.array([np.nan, 0.0]), np.array([[0.0, np.inf]]), np.array([], dtype=float)]
    for bad in bad_inputs:
        with pytest.raises(ValueError):
            fcn(bad)

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    sum_i N_i(ξ,η) = 1 for sample points (corners, mid-sides, centroid)."""
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]], dtype=float)
    (N, dN) = fcn(pts)
    assert N.shape[0] == pts.shape[0]
    sums = N[:, :, 0].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-12, rtol=0.0)

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies sum_i grad N_i(ξ,η) = (0,0) for sample points
    (corners, mid-sides, centroid)."""
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]], dtype=float)
    (N, dN) = fcn(pts)
    grads_sum = dN.sum(axis=1)
    assert np.allclose(grads_sum, 0.0, atol=1e-12, rtol=0.0)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    Assemble the 8x8 matrix M_{i,j} = N_i(node_j) and assert it equals the identity."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    (N, dN) = fcn(nodes)
    Nmat = N[:, :, 0].T
    assert Nmat.shape == (8, 8)
    assert np.allclose(Nmat, np.eye(8), atol=1e-12, rtol=0.0)

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    xi_vals = np.linspace(-1.0, 1.0, 5)
    eta_vals = np.linspace(-1.0, 1.0, 5)
    (XI, ETA) = np.meshgrid(xi_vals, eta_vals)
    pts = np.column_stack([XI.ravel(), ETA.ravel()])
    (N, dN) = fcn(pts)
    Ns = N[:, :, 0]
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    nx = nodes[:, 0]
    ny = nodes[:, 1]
    sx = pts[:, 0]
    sy = pts[:, 1]
    polys = [lambda x, y: np.ones_like(x), lambda x, y: x, lambda x, y: y, lambda x, y: x ** 2, lambda x, y: x * y, lambda x, y: y ** 2]
    tol = 1e-12
    for p in polys:
        nodal_vals = p(nx, ny)
        interp = Ns @ nodal_vals
        exact = p(sx, sy)
        err = np.max(np.abs(interp - exact))
        assert err < tol

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    xi_vals = np.linspace(-1.0, 1.0, 5)
    eta_vals = np.linspace(-1.0, 1.0, 5)
    (XI, ETA) = np.meshgrid(xi_vals, eta_vals)
    pts = np.column_stack([XI.ravel(), ETA.ravel()])
    (N, dN) = fcn(pts)
    dN_xi = dN[:, :, 0]
    dN_eta = dN[:, :, 1]
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    nx = nodes[:, 0]
    ny = nodes[:, 1]
    sx = pts[:, 0]
    sy = pts[:, 1]
    polys_and_grads = [(lambda x, y: np.ones_like(x), lambda x, y: np.zeros_like(x), lambda x, y: np.zeros_like(x)), (lambda x, y: x, lambda x, y: np.ones_like(x), lambda x, y: np.zeros_like(x)), (lambda x, y: y, lambda x, y: np.zeros_like(x), lambda x, y: np.ones_like(x)), (lambda x, y: x ** 2, lambda x, y: 2.0 * x, lambda x, y: np.zeros_like(x)), (lambda x, y: x * y, lambda x, y: y, lambda x, y: x), (lambda x, y: y ** 2, lambda x, y: np.zeros_like(x), lambda x, y: 2.0 * y)]
    tol = 1e-12
    for (p, pdx, pdy) in polys_and_grads:
        nodal_vals = p(nx, ny)
        grad_x_interp = dN_xi @ nodal_vals
        grad_y_interp = dN_eta @ nodal_vals
        grad_x_exact = pdx(sx, sy)
        grad_y_exact = pdy(sx, sy)
        errx = np.max(np.abs(grad_x_interp - grad_x_exact))
        erry = np.max(np.abs(grad_y_interp - grad_y_exact))
        assert errx < tol and erry < tol