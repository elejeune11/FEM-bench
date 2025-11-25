def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    with raises(ValueError):
        fcn(xi=[0, 0])
    with raises(ValueError):
        fcn(xi=np.array([0, 0, 0]))
    with raises(ValueError):
        fcn(xi=np.array([[0, 0], [0, np.nan]]))
    with raises(ValueError):
        fcn(xi=np.array([[0, 0], [np.inf, 0]]))

def test_partition_of_unity_tri6(fcn):
    xi = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    (N, _) = fcn(xi)
    assert np.allclose(np.sum(N, axis=1), 1.0)

def test_derivative_partition_of_unity_tri6(fcn):
    xi = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    (_, dN_dxi) = fcn(xi)
    assert np.allclose(np.sum(dN_dxi, axis=1), 0.0)

def test_kronecker_delta_at_nodes_tri6(fcn):
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    (N, _) = fcn(nodes)
    assert np.allclose(N, np.identity(6))

def test_value_completeness_tri6(fcn):
    xi = np.array([[0.2, 0.3], [0.6, 0.1], [0.1, 0.7]])
    p1 = lambda x, y: 2 * x + 3 * y - 1
    p2 = lambda x, y: x ** 2 + x * y + y ** 2
    (N1, _) = fcn(xi)
    v1 = np.array([p1(x, y) for (x, y) in xi])
    v2 = np.array([p2(x, y) for (x, y) in xi])
    assert np.allclose(np.dot(N1, np.array([p1(0, 0), p1(1, 0), p1(0, 1), p1(0.5, 0), p1(0, 0.5), p1(0.5, 0.5)]).reshape(-1, 1)), v1.reshape(-1, 1))
    assert np.allclose(np.dot(N1, np.array([p2(0, 0), p2(1, 0), p2(0, 1), p2(0.5, 0), p2(0, 0.5), p2(0.5, 0.5)]).reshape(-1, 1)), v2.reshape(-1, 1))

def test_gradient_completeness_tri6(fcn):
    xi = np.array([[0.2, 0.3], [0.6, 0.1], [0.1, 0.7]])
    p1 = lambda x, y: 2 * x + 3 * y - 1
    p2 = lambda x, y: x ** 2 + x * y + y ** 2
    (_, dN_dxi) = fcn(xi)
    grad1 = np.array([[2, 3], [2, 3], [2, 3]])
    grad2 = np.array([[2 * xi[0, 0] + xi[0, 1], 2 * xi[0, 1] + xi[0, 0]], [2 * xi[1, 0] + xi[1, 1], 2 * xi[1, 1] + xi[1, 0]], [2 * xi[2, 0] + xi[2, 1], 2 * xi[2, 1] + xi[2, 0]]])
    assert np.allclose(np.einsum('ijk,ik->ij', dN_dxi, np.array([2, 3])), grad1)
    assert np.allclose(np.einsum('ijk,ik->ij', dN_dxi, np.array([xi[:, 0] ** 2 + xi[:, 1] ** 2, xi[:, 0] * xi[:, 1]])), grad2)