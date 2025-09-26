def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    bad_inputs = ['not an array', [1, 2], np.array([1, 2, 3]), np.array([[1, 2], [3, 4], [5, 6]]), np.array([np.nan, 1]), np.array([1, np.inf])]
    for bad_input in bad_inputs:
        try:
            fcn(bad_input)
            assert False
        except ValueError:
            pass

def test_partition_of_unity_quad8(fcn):
    xi = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (N, _) = fcn(xi)
    assert np.allclose(np.sum(N, axis=1), 1)

def test_derivative_partition_of_unity_quad8(fcn):
    xi = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (_, dN_dxi) = fcn(xi)
    assert np.allclose(np.sum(dN_dxi, axis=1), 0)

def test_kronecker_delta_at_nodes_quad8(fcn):
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (N, _) = fcn(nodes)
    assert np.allclose(N, np.identity(8))

def test_value_completeness_quad8(fcn):
    xi = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    p1 = lambda x, y: 1 + 2 * x + 3 * y
    p2 = lambda x, y: 1 + 2 * x + 3 * y + 4 * x * y + 5 * x ** 2 + 6 * y ** 2
    nodal_values_p1 = np.array([p1(x, y) for (x, y) in xi]).reshape(-1, 1)
    (N, _) = fcn(xi)
    interp_p1 = N @ nodal_values_p1
    exact_p1 = np.array([p1(x, y) for (x, y) in xi]).reshape(-1, 1)
    assert np.allclose(interp_p1, exact_p1)
    nodal_values_p2 = np.array([p2(x, y) for (x, y) in xi]).reshape(-1, 1)
    (N, _) = fcn(xi)
    interp_p2 = N @ nodal_values_p2
    exact_p2 = np.array([p2(x, y) for (x, y) in xi]).reshape(-1, 1)
    assert np.allclose(interp_p2, exact_p2)

def test_gradient_completeness_quad8(fcn):
    xi = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    p1 = lambda x, y: 1 + 2 * x + 3 * y
    p2 = lambda x, y: 1 + 2 * x + 3 * y + 4 * x * y + 5 * x ** 2 + 6 * y ** 2
    grad_p1 = lambda x, y: np.array([2, 3])
    grad_p2 = lambda x, y: np.array([2 + 4 * y + 10 * x, 3 + 4 * x + 12 * y])
    nodal_values_p1 = np.array([p1(x, y) for (x, y) in xi]).reshape(-1, 1)
    (N, dN_dxi) = fcn(xi)
    interp_grad_p1 = (dN_dxi @ nodal_values_p1).reshape(-1, 2)
    exact_grad_p1 = np.array([grad_p1(x, y) for (x, y) in xi])
    assert np.allclose(interp_grad_p1, exact_grad_p1)
    nodal_values_p2 = np.array([p2(x, y) for (x, y) in xi]).reshape(-1, 1)
    (N, dN_dxi) = fcn(xi)
    interp_grad_p2 = (dN_dxi @ nodal_values_p2).reshape(-1, 2)
    exact_grad_p2 = np.array([grad_p2(x, y) for (x, y) in xi])
    assert np.allclose(interp_grad_p2, exact_grad_p2)