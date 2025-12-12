def test_multi_element_core_correctness_assembly(fcn):
    """
    Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
    1) zero displacement produces a zero matrix,
    2) the assembled matrix is symmetric,
    3) scaling displacements scales K_g linearly,
    4) superposition holds for independent displacement states, and
    5) element order does not affect the assembled result.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667, 'local_z': np.array([0.0, 0.0, 1.0])}, {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667, 'local_z': np.array([0.0, 0.0, 1.0])}]
    u_zero = np.zeros(18)
    K_zero = fcn(node_coords, elements, u_zero)
    assert K_zero.shape == (18, 18), 'Global stiffness matrix should be 18x18 for 3 nodes'
    assert np.allclose(K_zero, 0.0), 'Zero displacement should produce zero geometric stiffness'
    u_nonzero = np.random.randn(18) * 0.01
    K_nonzero = fcn(node_coords, elements, u_nonzero)
    assert np.allclose(K_nonzero, K_nonzero.T, atol=1e-10), 'Geometric stiffness matrix should be symmetric'
    scale = 2.0
    u_scaled = u_nonzero * scale
    K_scaled = fcn(node_coords, elements, u_scaled)
    assert np.allclose(K_scaled, K_nonzero * scale, atol=1e-10), 'Scaling displacements should scale K_g linearly'
    u1 = np.random.randn(18) * 0.001
    u2 = np.random.randn(18) * 0.001
    K1 = fcn(node_coords, elements, u1)
    K2 = fcn(node_coords, elements, u2)
    K_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_sum, K1 + K2, atol=1e-08), 'Superposition should hold for independent displacement states'
    elements_reversed = list(reversed(elements))
    K_reversed = fcn(node_coords, elements_reversed, u_nonzero)
    assert np.allclose(K_nonzero, K_reversed, atol=1e-10), 'Element order should not affect the assembled result'

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667, 'local_z': np.array([0.0, 0.0, 1.0])}, {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667, 'local_z': np.array([0.0, 0.0, 1.0])}]
    u_global = np.random.randn(18) * 0.001
    K_orig = fcn(node_coords, elements, u_global)
    theta = np.pi / 6
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    node_coords_rot = node_coords @ R.T
    u_global_rot = np.zeros_like(u_global)
    for i in range(3):
        u_global_rot[6 * i:6 * i + 3] = u_global[6 * i:6 * i + 3] @ R.T
        u_global_rot[6 * i + 3:6 * i + 6] = u_global[6 * i + 3:6 * i + 6] @ R.T
    elements_rot = []
    for elem in elements:
        elem_rot = elem.copy()
        if 'local_z' in elem and elem['local_z'] is not None:
            elem_rot['local_z'] = elem['local_z'] @ R.T
        elements_rot.append(elem_rot)
    K_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
    T = np.zeros((18, 18))
    for i in range(3):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    K_transformed = T @ K_orig @ T.T
    assert np.allclose(K_rot, K_transformed, atol=1e-08), 'Frame objectivity should be satisfied: K_g^rot ≈ T K_g T^T'