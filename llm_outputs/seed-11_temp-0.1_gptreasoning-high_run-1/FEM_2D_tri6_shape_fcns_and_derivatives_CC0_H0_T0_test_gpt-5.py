def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """
    D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
or (n,2) with finite values. Invalid inputs should raise ValueError.
This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
that ValueError is raised.
    """
    invalid_types = [[0.2, 0.3], (0.2, 0.3), None, '0.2,0.3', 0.2]
    for inval in invalid_types:
        with pytest.raises(ValueError):
            fcn(inval)
    wrong_shapes = [np.array([0.1]), np.array([0.1, 0.2, 0.3]), np.array([[0.1, 0.2, 0.3]]), np.array([[0.1], [0.2]]), np.array([[[0.1, 0.2]]])]
    for inval in wrong_shapes:
        with pytest.raises(ValueError):
            fcn(inval)
    non_finite_inputs = [np.array([np.nan, 0.0]), np.array([np.inf, 0.0]), np.array([[0.1, 0.2], [np.nan, 0.3]]), np.array([[0.1, np.inf], [0.2, 0.3]])]
    for inval in non_finite_inputs:
        with pytest.raises(ValueError):
            fcn(inval)

def test_partition_of_unity_tri6(fcn):
    """
    Shape functions on a triangle must satisfy the partition of unity:
∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
This test evaluates ∑ N_i at well considered sample points and ensures 
that the sum equals 1 within tight tolerance.
    """
    xi = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.2, 0.3], [0.3, 0.2], [0.1, 0.6], [0.6, 0.1]], dtype=float)
    N, dN = fcn(xi)
    sumN = np.sum(N, axis=1)
    sumN = np.squeeze(sumN)
    assert np.allclose(sumN, np.ones_like(sumN), rtol=0.0, atol=1e-13)

def test_derivative_partition_of_unity_tri6(fcn):
    """
    Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    xi = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.2, 0.3], [0.3, 0.2], [0.1, 0.6], [0.6, 0.1]], dtype=float)
    N, dN = fcn(xi)
    grad_sum = np.sum(dN, axis=1)
    assert np.allclose(grad_sum, np.zeros_like(grad_sum), rtol=0.0, atol=1e-13)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """
    For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
This test evaluates N at each reference node location and assembles a 6×6 matrix whose
(i,j) entry is N_i at node_j. This should equal the identity within tight tolerance.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    N, dN = fcn(nodes)
    if N.ndim == 3:
        M = N[:, :, 0]
    else:
        M = N
    I = np.eye(6, dtype=float)
    assert np.allclose(M, I, rtol=0.0, atol=1e-13)

def test_value_completeness_tri6(fcn):
    """
    Check that quadratic (P2) triangle shape functions exactly reproduce
degree-1 and degree-2 polynomials. Nodal values are set from the exact
polynomial, the field is interpolated at sample points, and the maximum
error is verified to be nearly zero.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    samples = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.2, 0.3], [0.3, 0.2], [0.1, 0.6], [0.6, 0.1]], dtype=float)

    def eval_poly(coeffs, pts):
        A, B, C, D, E, F = coeffs
        x = pts[:, 0]
        y = pts[:, 1]
        return A * x ** 2 + B * y ** 2 + C * x * y + D * x + E * y + F
    polys = [(0.0, 0.0, 0.0, 0.0, 0.0, 3.7), (0.0, 0.0, 0.0, 1.2, -0.7, 0.33), (0.5, -0.4, 0.3, -0.2, 0.1, 0.05)]
    for coeffs in polys:
        v_nodes = eval_poly(coeffs, nodes)
        N_samp, _ = fcn(samples)
        interp_vals = np.tensordot(N_samp, v_nodes, axes=([1], [0]))
        interp_vals = np.squeeze(interp_vals)
        exact_vals = eval_poly(coeffs, samples)
        err = np.max(np.abs(interp_vals - exact_vals))
        assert err < 1e-12

def test_gradient_completeness_tri6(fcn):
    """
    Check that P2 triangle shape functions reproduce the exact gradient for
degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
values and compared with the analytic gradient at sample points, with
maximum error verified to be nearly zero.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    samples = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.2, 0.3], [0.3, 0.2], [0.1, 0.6], [0.6, 0.1]], dtype=float)

    def grad_poly(coeffs, pts):
        A, B, C, D, E, _ = coeffs
        x = pts[:, 0]
        y = pts[:, 1]
        gx = 2 * A * x + C * y + D
        gy = 2 * B * y + C * x + E
        return np.stack([gx, gy], axis=1)
    polys = [(0.0, 0.0, 0.0, 0.0, 0.0, 3.7), (0.0, 0.0, 0.0, 1.2, -0.7, 0.33), (0.5, -0.4, 0.3, -0.2, 0.1, 0.05)]
    for coeffs in polys:
        v_nodes = nodes[:, 0] * 0.0
        A, B, C, D, E, F = coeffs
        x_nodes = nodes[:, 0]
        y_nodes = nodes[:, 1]
        v_nodes = A * x_nodes ** 2 + B * y_nodes ** 2 + C * x_nodes * y_nodes + D * x_nodes + E * y_nodes + F
        _, dN_samp = fcn(samples)
        grad_interp = np.tensordot(dN_samp, v_nodes, axes=([1], [0]))
        exact_grad = grad_poly(coeffs, samples)
        err = np.max(np.abs(grad_interp - exact_grad))
        assert err < 1e-12