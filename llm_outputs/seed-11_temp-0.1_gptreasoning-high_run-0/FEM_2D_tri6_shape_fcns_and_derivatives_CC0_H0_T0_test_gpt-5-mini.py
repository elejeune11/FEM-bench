def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised."""
    invalid_inputs = [[0.1, 0.2], (0.1, 0.2), np.array([0.1, 0.2, 0.3]), np.array([[0.1], [0.2]]), np.array([[0.1, 0.2, 0.3]]), np.array([np.nan, 0.0]), np.array([[0.1, np.inf]]), np.array([[0.5, 0.6], [np.nan, 0.0]])]
    for inp in invalid_inputs:
        with pytest.raises(ValueError):
            fcn(inp)

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity:
    sum_i N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates sum N_i at well considered sample points and ensures
    that the sum equals 1 within tight tolerance."""
    pts = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.5, 0.0], [0.0, 0.5], [1 / 3, 1 / 3], [0.25, 0.1], [0.1, 0.2], [0.2, 0.7]], dtype=float)
    (N, dN) = fcn(pts)
    assert isinstance(N, np.ndarray) and isinstance(dN, np.ndarray)
    Nvals = N[:, :, 0]
    sums = Nvals.sum(axis=1)
    assert np.allclose(sums, np.ones_like(sums), atol=1e-12, rtol=0)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies sum_i grad N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates sum grad N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    pts = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.25, 0.25], [1 / 3, 1 / 3], [0.1, 0.2], [0.2, 0.1]], dtype=float)
    (N, dN) = fcn(pts)
    sums = dN.sum(axis=1)
    assert np.allclose(sums, np.zeros_like(sums), atol=1e-12, rtol=0)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    (N, dN) = fcn(nodes)
    Nmat = N[:, :, 0].T
    assert Nmat.shape == (6, 6)
    assert np.allclose(Nmat, np.eye(6), atol=1e-12, rtol=0)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    rng = np.random.RandomState(0)
    r = rng.rand(200, 2)
    mask = r.sum(axis=1) <= 1.0
    pts = r[mask][:50]

    def p_lin(x, y):
        return 2.0 - 3.0 * x + 4.0 * y

    def p_quad(x, y):
        return -1.5 + 2.3 * x - 0.7 * y + 1.4 * x * y + 0.9 * x * x - 0.6 * y * y
    for poly in (p_lin, p_quad):
        nodal_vals = poly(nodes[:, 0], nodes[:, 1])
        (Nvals, dNvals) = fcn(pts)
        Nmat = Nvals[:, :, 0]
        interp = Nmat.dot(nodal_vals)
        exact = poly(pts[:, 0], pts[:, 1])
        err = np.max(np.abs(interp - exact))
        assert err <= 1e-12

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    rng = np.random.RandomState(1)
    r = rng.rand(300, 2)
    mask = r.sum(axis=1) <= 1.0
    pts = r[mask][:60]
    (a0, a1, a2) = (2.0, -3.0, 4.0)

    def p_lin(x, y):
        return a0 + a1 * x + a2 * y

    def grad_lin(x, y):
        gx = np.full_like(x, a1)
        gy = np.full_like(y, a2)
        return np.column_stack((gx, gy))
    A = -1.5
    B = 2.3
    C = -0.7
    D = 1.4
    E = 0.9
    F = -0.6

    def p_quad(x, y):
        return A + B * x + C * y + D * x * y + E * x * x + F * y * y

    def grad_quad(x, y):
        gx = B + D * y + 2.0 * E * x
        gy = C + D * x + 2.0 * F * y
        return np.column_stack((gx, gy))
    for (poly, grad_fn) in ((p_lin, grad_lin), (p_quad, grad_quad)):
        nodal_vals = poly(nodes[:, 0], nodes[:, 1])
        (Nvals, dNvals) = fcn(pts)
        grad_rec = np.tensordot(dNvals, nodal_vals, axes=([1], [0]))
        analytic = grad_fn(pts[:, 0], pts[:, 1])
        err = np.max(np.abs(grad_rec - analytic))
        assert err <= 1e-12