def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    import sys
    import types
    module = types.ModuleType('mock_module')
    module.local_elastic_stiffness_matrix_3D_beam = local_elastic_stiffness_matrix_3D_beam
    module.beam_transformation_matrix_3D = beam_transformation_matrix_3D
    sys.modules['mock_module'] = module
    fcn.__globals__['local_elastic_stiffness_matrix_3D_beam'] = local_elastic_stiffness_matrix_3D_beam
    fcn.__globals__['beam_transformation_matrix_3D'] = beam_transformation_matrix_3D
    K = fcn(node_coords, elements)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T, rtol=1e-12)
    assert not np.allclose(K, np.zeros_like(K))
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    K = fcn(node_coords, elements)
    assert K.shape == (18, 18)
    assert np.allclose(K, K.T, rtol=1e-12)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 2, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    K = fcn(node_coords, elements)
    assert K.shape == (18, 18)
    assert np.allclose(K, K.T, rtol=1e-12)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 3, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    K = fcn(node_coords, elements)
    assert K.shape == (24, 24)
    assert np.allclose(K, K.T, rtol=1e-12)
    for i in range(4):
        block = K[6 * i:6 * i + 6, 6 * i:6 * i + 6]
        assert not np.allclose(block, np.zeros((6, 6)))