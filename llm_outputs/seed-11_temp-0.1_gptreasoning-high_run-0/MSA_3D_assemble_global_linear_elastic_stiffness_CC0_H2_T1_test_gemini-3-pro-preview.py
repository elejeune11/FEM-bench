def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop.
    """

    def mock_local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, I_y, I_z, J):
        k = np.zeros((12, 12))
        stiffness_factor = E * A / L
        np.fill_diagonal(k, stiffness_factor)
        k[0, 6] = -stiffness_factor
        k[6, 0] = -stiffness_factor
        return k

    def mock_beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        return np.eye(12)
    fcn.__globals__['local_elastic_stiffness_matrix_3D_beam'] = mock_local_elastic_stiffness_matrix_3D_beam
    fcn.__globals__['beam_transformation_matrix_3D'] = mock_beam_transformation_matrix_3D
    props = {'E': 1000.0, 'nu': 0.3, 'A': 0.1, 'I_y': 1.0, 'I_z': 1.0, 'J': 1.0}
    nodes_1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elems_1 = [dict(props, node_i=0, node_j=1)]
    K1 = fcn(nodes_1, elems_1)
    assert K1.shape == (12, 12), 'Single element matrix shape mismatch'
    assert np.allclose(K1, K1.T), 'Single element matrix is not symmetric'
    assert np.isclose(K1[0, 0], 100.0)
    assert np.isclose(K1[0, 6], -100.0)
    nodes_2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elems_2 = [dict(props, node_i=0, node_j=1), dict(props, node_i=1, node_j=2)]
    K2 = fcn(nodes_2, elems_2)
    assert K2.shape == (18, 18), 'Linear chain matrix shape mismatch'
    assert np.allclose(K2, K2.T), 'Linear chain matrix is not symmetric'
    assert np.isclose(K2[6, 6], 200.0)
    nodes_3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    elems_3 = [dict(props, node_i=0, node_j=1), dict(props, node_i=1, node_j=2), dict(props, node_i=2, node_j=0)]
    K3 = fcn(nodes_3, elems_3)
    assert K3.shape == (18, 18), 'Triangle loop matrix shape mismatch'
    assert np.allclose(K3, K3.T), 'Triangle loop matrix is not symmetric'
    assert np.any(K3[0:6, 12:18]), 'Stiffness block (0,2) is zero; loop not closed?'
    assert np.any(K3[12:18, 0:6]), 'Stiffness block (2,0) is zero; loop not closed?'
    nodes_4 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elems_4 = [dict(props, node_i=0, node_j=1), dict(props, node_i=1, node_j=2), dict(props, node_i=2, node_j=3), dict(props, node_i=3, node_j=0)]
    K4 = fcn(nodes_4, elems_4)
    assert K4.shape == (24, 24), 'Square loop matrix shape mismatch'
    assert np.allclose(K4, K4.T), 'Square loop matrix is not symmetric'
    assert np.any(K4[0:6, 18:24]), 'Square loop closure (0,3) missing'