def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    with pytest.raises(ValueError):
        fcn([0, 0])
    with pytest.raises(ValueError):
        fcn(np.array([0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0, 0, 0]]))
    with pytest.raises(ValueError):
        fcn(np.array([[np.nan, 0]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0, np.inf]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0, 0], [np.nan, 0]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0, 0], [0, np.inf]]))

def test_partition_of_unity_quad8(fcn):
    points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (N, _) = fcn(points)
    sums = np.sum(N.squeeze(), axis=1)
    assert np.allclose(sums, 1.0, atol=1e-12)

def test_derivative_partition_of_unity_quad8(fcn):
    points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (_, dN_dxi) = fcn(points)
    grad_sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(grad_sums, 0.0, atol=1e-12)

def test_kronecker_delta_at_nodes_quad8(fcn):
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (N, _) = fcn(nodes)
    N_matrix = N.squeeze()
    identity = np.eye(8)
    assert np.allclose(N_matrix, identity, atol=1e-12)

def test_value_completeness_quad8(fcn):

    def poly1(x, y):
        return 1 + 2 * x + 3 * y

    def poly2(x, y):
        return 1 + 2 * x + 3 * y + 4 * x ** 2 + 5 * x * y + 6 * y ** 2
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    nodal_vals1 = poly1(nodes[:, 0], nodes[:, 1])
    (N, _) = fcn(nodes)
    interp1 = np.einsum('ij,i->j', N.squeeze(), nodal_vals1)
    exact1 = poly1(nodes[:, 0], nodes[:, 1])
    assert np.allclose(interp1, exact1, atol=1e-12)
    nodal_vals2 = poly2(nodes[:, 0], nodes[:, 1])
    interp2 = np.einsum('ij,i->j', N.squeeze(), nodal_vals2)
    exact2 = poly2(nodes[:, 0], nodes[:, 1])
    assert np.allclose(interp2, exact2, atol=1e-12)
    test_points = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0, 0]])
    (N_test, _) = fcn(test_points)
    interp1_test = np.einsum('ij,j->i', N_test.squeeze(), nodal_vals1)
    exact1_test = poly1(test_points[:, 0], test_points[:, 1])
    assert np.allclose(interp1_test, exact1_test, atol=1e-12)
    interp2_test = np.einsum('ij,j->i', N_test.squeeze(), nodal_vals2)
    exact2_test = poly2(test_points[:, 0], test_points[:, 1])
    assert np.allclose(interp2_test, exact2_test, atol=1e-12)

def test_gradient_completeness_quad8(fcn):

    def poly1(x, y):
        return 1 + 2 * x + 3 * y

    def grad_poly1(x, y):
        return np.stack([2 * np.ones_like(x), 3 * np.ones_like(y)], axis=-1)

    def poly2(x, y):
        return 1 + 2 * x + 3 * y + 4 * x ** 2 + 5 * x * y + 6 * y ** 2

    def grad_poly2(x, y):
        return np.stack([2 + 8 * x + 5 * y, 3 + 5 * x + 12 * y], axis=-1)
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    nodal_vals1 = poly1(nodes[:, 0], nodes[:, 1])
    nodal_vals2 = poly2(nodes[:, 0], nodes[:, 1])
    (_, dN_dxi) = fcn(nodes)
    grad_interp1 = np.einsum('ijk,i->jk', dN_dxi, nodal_vals1)
    grad_exact1 = grad_poly1(nodes[:, 0], nodes[:, 1])
    assert np.allclose(grad_interp1, grad_exact1, atol=1e-12)
    grad_interp2 = np.einsum('ijk,i->jk', dN_dxi, nodal_vals2)
    grad_exact2 = grad_poly2(nodes[:, 0], nodes[:, 1])
    assert np.allclose(grad_interp2, grad_exact2, atol=1e-12)
    test_points = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0, 0]])
    (_, dN_dxi_test) = fcn(test_points)
    grad_interp1_test = np.einsum('ijk,i->jk', dN_dxi_test, nodal_vals1)
    grad_exact1_test = grad_poly1(test_points[:, 0], test_points[:, 1])
    assert np.allclose(grad_interp1_test, grad_exact1_test, atol=1e-12)
    grad_interp2_test = np.einsum('ijk,i->jk', dN_dxi_test, nodal_vals2)
    grad_exact2_test = grad_poly2(test_points[:, 0], test_points[:, 1])
    assert np.allclose(grad_interp2_test, grad_exact2_test, atol=1e-12)