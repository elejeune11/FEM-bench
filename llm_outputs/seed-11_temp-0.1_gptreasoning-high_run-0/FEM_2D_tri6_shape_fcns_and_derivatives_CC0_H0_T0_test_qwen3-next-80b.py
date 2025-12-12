def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    with pytest.raises(ValueError):
        fcn([0.5, 0.5])
    with pytest.raises(ValueError):
        fcn(np.array([0.5]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.5, 0.5], [0.3, 0.4], [0.2, 0.1]]).reshape(3, 2, 1))
    with pytest.raises(ValueError):
        fcn(np.array([[np.nan, 0.5]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.5, np.inf]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.5, 0.5], [0.3, np.nan]]))

def test_partition_of_unity_tri6(fcn):
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.25, 0.25], [0.7, 0.2], [0.2, 0.7]])
    (N, _) = fcn(points)
    sums = np.sum(N[:, :, 0], axis=1)
    assert np.allclose(sums, 1.0, atol=1e-12)

def test_derivative_partition_of_unity_tri6(fcn):
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.25, 0.25], [0.7, 0.2], [0.2, 0.7]])
    (_, dN_dxi) = fcn(points)
    grad_sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(grad_sums, 0.0, atol=1e-12)

def test_kronecker_delta_at_nodes_tri6(fcn):
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    N_all = []
    for node in nodes:
        (N, _) = fcn(node.reshape(1, 2))
        N_all.append(N[0, :, 0])
    N_matrix = np.array(N_all)
    identity = np.eye(6)
    assert np.allclose(N_matrix, identity, atol=1e-12)

def test_value_completeness_tri6(fcn):

    def exact_poly1(xi, eta):
        return 1.0 + 2.0 * xi + 3.0 * eta

    def exact_poly2(xi, eta):
        return 1.0 + 2.0 * xi + 3.0 * eta + 4.0 * xi ** 2 + 5.0 * eta ** 2 + 6.0 * xi * eta
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    vals1 = np.array([exact_poly1(xi, eta) for (xi, eta) in nodes])
    (N, _) = fcn(nodes)
    interp_vals1 = np.dot(N[:, :, 0], vals1)
    test_points = np.array([[0.1, 0.1], [0.3, 0.2], [0.2, 0.4], [0.6, 0.1], [0.1, 0.6]])
    (N_test, _) = fcn(test_points)
    computed1 = np.dot(N_test[:, :, 0], vals1)
    exact1 = np.array([exact_poly1(xi, eta) for (xi, eta) in test_points])
    assert np.allclose(computed1, exact1, atol=1e-10)
    vals2 = np.array([exact_poly2(xi, eta) for (xi, eta) in nodes])
    interp_vals2 = np.dot(N[:, :, 0], vals2)
    computed2 = np.dot(N_test[:, :, 0], vals2)
    exact2 = np.array([exact_poly2(xi, eta) for (xi, eta) in test_points])
    assert np.allclose(computed2, exact2, atol=1e-10)

def test_gradient_completeness_tri6(fcn):

    def exact_poly1(xi, eta):
        return 1.0 + 2.0 * xi + 3.0 * eta

    def exact_poly2(xi, eta):
        return 1.0 + 2.0 * xi + 3.0 * eta + 4.0 * xi ** 2 + 5.0 * eta ** 2 + 6.0 * xi * eta

    def grad_poly1(xi, eta):
        return np.array([2.0, 3.0])

    def grad_poly2(xi, eta):
        return np.array([2.0 + 8.0 * xi + 6.0 * eta, 3.0 + 10.0 * eta + 6.0 * xi])
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    vals1 = np.array([exact_poly1(xi, eta) for (xi, eta) in nodes])
    (N, dN_dxi) = fcn(nodes)
    grad_interp1 = np.einsum('ij,ikj->ik', vals1, dN_dxi)
    test_points = np.array([[0.1, 0.1], [0.3, 0.2], [0.2, 0.4], [0.6, 0.1], [0.1, 0.6]])
    (_, dN_test) = fcn(test_points)
    computed_grad1 = np.einsum('ij,ikj->ik', vals1, dN_test)
    exact_grad1 = np.array([grad_poly1(xi, eta) for (xi, eta) in test_points])
    assert np.allclose(computed_grad1, exact_grad1, atol=1e-10)
    vals2 = np.array([exact_poly2(xi, eta) for (xi, eta) in nodes])
    grad_interp2 = np.einsum('ij,ikj->ik', vals2, dN_dxi)
    computed_grad2 = np.einsum('ij,ikj->ik', vals2, dN_test)
    exact_grad2 = np.array([grad_poly2(xi, eta) for (xi, eta) in test_points])
    assert np.allclose(computed_grad2, exact_grad2, atol=1e-10)