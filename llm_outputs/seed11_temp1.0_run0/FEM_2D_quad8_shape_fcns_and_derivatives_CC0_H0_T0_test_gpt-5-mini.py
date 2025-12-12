def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """
    The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised.
    """
    bad_inputs = [[0.0, 0.0], np.array([0.0]), np.array([0.0, 0.0, 0.0]), np.array([[0.0], [0.0]]), np.array([[0.0, 0.0, 0.0]]), np.array([[np.nan, 0.0]]), np.array([0.0, np.inf]), np.array([[0.0, np.nan], [0.0, 0.0]])]
    for inp in bad_inputs:
        with pytest.raises(ValueError):
            fcn(inp)

def test_partition_of_unity_quad8(fcn):
    """
    Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance.
    """
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    (N, dN) = fcn(pts)
    N = np.asarray(N)
    assert N.shape[0] == pts.shape[0]
    Nmat = N.squeeze(-1)
    sums = Nmat.sum(axis=1)
    assert np.allclose(sums, np.ones_like(sums), atol=1e-12, rtol=0.0)

def test_derivative_partition_of_unity_quad8(fcn):
    """
    Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    (N, dN) = fcn(pts)
    dN = np.asarray(dN)
    sums = dN.sum(axis=1)
    assert np.allclose(sums, np.zeros_like(sums), atol=1e-12, rtol=0.0)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """
    For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (N, dN) = fcn(nodes)
    N = np.asarray(N).squeeze(-1)
    M = N.T
    I = np.eye(8)
    assert np.allclose(M, I, atol=1e-12, rtol=0.0)

def test_value_completeness_quad8(fcn):
    """
    Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    grid = np.linspace(-1.0, 1.0, 5)
    pts = np.array([[x, y] for y in grid for x in grid])
    rng = np.random.default_rng(12345)
    (a, b, c) = rng.uniform(-2.0, 2.0, size=3)

    def poly_lin(xi, eta):
        return a + b * xi + c * eta
    coeffs = rng.uniform(-2.0, 2.0, size=6)

    def poly_quad(xi, eta):
        return coeffs[0] + coeffs[1] * xi + coeffs[2] * eta + coeffs[3] * xi * xi + coeffs[4] * xi * eta + coeffs[5] * eta * eta
    for poly in (poly_lin, poly_quad):
        nodal_vals = np.array([poly(x, y) for (x, y) in nodes])
        (N_pts, dN_pts) = fcn(pts)
        N_pts = np.asarray(N_pts).squeeze(-1)
        interp = N_pts.dot(nodal_vals)
        exact = np.array([poly(x, y) for (x, y) in pts])
        err = np.abs(interp - exact)
        assert np.max(err) < 1e-10

def test_gradient_completeness_quad8(fcn):
    """
    Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    grid = np.linspace(-1.0, 1.0, 5)
    pts = np.array([[x, y] for y in grid for x in grid])
    rng = np.random.default_rng(67890)
    (a, b, c) = rng.uniform(-2.0, 2.0, size=3)

    def poly_lin(xi, eta):
        return a + b * xi + c * eta

    def grad_lin(xi, eta):
        return np.array([b, c])
    coeffs = rng.uniform(-2.0, 2.0, size=6)

    def poly_quad(xi, eta):
        return coeffs[0] + coeffs[1] * xi + coeffs[2] * eta + coeffs[3] * xi * xi + coeffs[4] * xi * eta + coeffs[5] * eta * eta

    def grad_quad(xi, eta):
        dfdxi = coeffs[1] + 2.0 * coeffs[3] * xi + coeffs[4] * eta
        dfdeta = coeffs[2] + coeffs[4] * xi + 2.0 * coeffs[5] * eta
        return np.array([dfdxi, dfdeta])
    for (poly, grad) in ((poly_lin, grad_lin), (poly_quad, grad_quad)):
        nodal_vals = np.array([poly(x, y) for (x, y) in nodes])
        (N_pts, dN_pts) = fcn(pts)
        dN_pts = np.asarray(dN_pts)
        grad_recon = np.einsum('nij,j->ni', dN_pts, nodal_vals)
        grad_exact = np.array([grad(x, y) for (x, y) in pts])
        err = np.linalg.norm(grad_recon - grad_exact, axis=1)
        assert np.max(err) < 1e-10