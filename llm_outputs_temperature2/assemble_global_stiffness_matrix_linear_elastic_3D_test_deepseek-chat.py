def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop.
    """
    node_coords = np.array([[0, 0, 0], [1, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}]
    K = fcn(node_coords, elements)
    n_nodes = node_coords.shape[0]
    expected_shape = (6 * n_nodes, 6 * n_nodes)
    assert K.shape == expected_shape, f'Expected shape {expected_shape}, got {K.shape}'
    assert np.allclose(K, K.T), 'Stiffness matrix should be symmetric'
    element_block = K[0:12, 0:12]
    assert not np.allclose(element_block, 0), "Element should contribute to its nodes' DOFs"
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}]
    K = fcn(node_coords, elements)
    n_nodes = node_coords.shape[0]
    expected_shape = (6 * n_nodes, 6 * n_nodes)
    assert K.shape == expected_shape, f'Expected shape {expected_shape}, got {K.shape}'
    assert np.allclose(K, K.T), 'Stiffness matrix should be symmetric'
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 2, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}]
    K = fcn(node_coords, elements)
    n_nodes = node_coords.shape[0]
    expected_shape = (6 * n_nodes, 6 * n_nodes)
    assert K.shape == expected_shape, f'Expected shape {expected_shape}, got {K.shape}'
    assert np.allclose(K, K.T), 'Stiffness matrix should be symmetric'
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 3, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}]
    K = fcn(node_coords, elements)
    n_nodes = node_coords.shape[0]
    expected_shape = (6 * n_nodes, 6 * n_nodes)
    assert K.shape == expected_shape, f'Expected shape {expected_shape}, got {K.shape}'
    assert np.allclose(K, K.T), 'Stiffness matrix should be symmetric'

def test_assemble_global_stiffness_matrix_single_element(fcn):
    """Tests the assembly for a single beam element with known properties."""
    node_coords = np.array([[0, 0, 0], [2, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}]
    K = fcn(node_coords, elements)
    assert K.shape == (12, 12), 'Single element should produce 12x12 matrix'
    assert np.allclose(K, K.T), 'Matrix should be symmetric'
    eigenvalues = np.linalg.eigvalsh(K)
    assert np.all(eigenvalues >= -1e-10), 'Stiffness matrix should be positive semi-definite'
    zero_eigenvalues = np.sum(np.abs(eigenvalues) < 1e-06)
    assert zero_eigenvalues == 6, f'Expected 6 rigid body modes, found {zero_eigenvalues}'

def test_assemble_global_stiffness_matrix_connectivity(fcn):
    """Tests that elements correctly connect nodes and contribute to appropriate DOFs."""
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}]
    K = fcn(node_coords, elements)
    block_0_1 = K[0:12, 0:12]
    assert not np.allclose(block_0_1, 0), 'Element 1 should contribute to nodes 0-1 block'
    block_1_2 = K[6:18, 6:18]
    assert not np.allclose(block_1_2, 0), 'Element 2 should contribute to nodes 1-2 block'
    block_0_2 = K[0:6, 12:18]
    assert np.allclose(block_0_2, 0), 'Nodes 0 and 2 should not be directly connected'

def test_assemble_global_stiffness_matrix_material_properties(fcn):
    """Tests that different material properties affect the stiffness matrix appropriately."""
    node_coords = np.array([[0, 0, 0], [1, 0, 0]])
    elements_high = [{'node_i': 0, 'node_j': 1, 'E': 400000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}]
    elements_low = [{'node_i': 0, 'node_j': 1, 'E': 100000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}]
    K_high = fcn(node_coords, elements_high)
    K_low = fcn(node_coords, elements_low)
    assert np.linalg.norm(K_high) > np.linalg.norm(K_low), 'Higher E should produce stiffer matrix'
    ratio = np.linalg.norm(K_high) / np.linalg.norm(K_low)
    assert 3.5 < ratio < 4.5, f'Stiffness ratio should be about 4:1, got {ratio}'

def test_assemble_global_stiffness_matrix_optional_local_z(fcn):
    """Tests that the optional local_z parameter is handled correctly."""
    node_coords = np.array([[0, 0, 0], [1, 0, 0]])
    elements_no_z = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}]
    elements_with_z = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': [0, 0, 1]}]
    K_no_z = fcn(node_coords, elements_no_z)
    K_with_z = fcn(node_coords, elements_with_z)
    assert K_no_z.shape == (12, 12), 'Matrix shape should be correct without local_z'
    assert K_with_z.shape == (12, 12), 'Matrix shape should be correct with local_z'
    assert np.allclose(K_no_z, K_no_z.T), 'Matrix should be symmetric without local_z'
    assert np.allclose(K_with_z, K_with_z.T), 'Matrix should be symmetric with local_z'

def test_assemble_global_stiffness_matrix_3D_geometry(fcn):
    """Tests assembly with 3D geometry (non-coplanar nodes)."""
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0], [0.5, 0.288, 0.816]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 2, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 0, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 1, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}]
    K = fcn(node_coords, elements)
    n_nodes = node_coords.shape[0]
    expected_shape = (6 * n_nodes, 6 * n_nodes)
    assert K.shape == expected_shape, f'Expected shape {expected_shape}, got {K.shape}'
    assert np.allclose(K, K.T), '3D geometry matrix should be symmetric'
    eigenvalues = np.linalg.eigvalsh(K)
    assert np.all(eigenvalues >= -1e-10), '3D stiffness matrix should be positive semi-definite'