def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result."""
    import numpy as np
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.1}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 0.1}]
    u_zero = np.zeros(18)
    K_zero = fcn(node_coords, elements, u_zero)
    assert K_zero.shape == (18, 18)
    assert np.allclose(K_zero, 0.0)
    u_test = np.random.rand(18) * 0.1
    K_test = fcn(node_coords, elements, u_test)
    assert np.allclose(K_test, K_test.T)
    scale = 2.5
    u_scaled = scale * u_test
    K_scaled = fcn(node_coords, elements, u_scaled)
    K_expected = scale * K_test
    assert np.allclose(K_scaled, K_expected)
    u_test2 = np.random.rand(18) * 0.1
    K_test2 = fcn(node_coords, elements, u_test2)
    K_sum = fcn(node_coords, elements, u_test + u_test2)
    assert np.allclose(K_sum, K_test + K_test2)
    elements_reordered = [elements[1], elements[0]]
    K_reordered = fcn(node_coords, elements_reordered, u_test)
    assert np.allclose(K_test, K_reordered)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz]."""
    import numpy as np
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.2]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.1, 'local_z': [0, 0, 1]}]
    u_global = np.array([0.01, 0.02, 0.005, 0.001, 0.002, 0.001, 0.015, 0.01, 0.008, 0.002, 0.001, 0.003])
    K_orig = fcn(node_coords, elements, u_global)
    theta = np.pi / 6
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    node_coords_rot = (R @ node_coords.T).T
    local_z_rot = R @ np.array([0, 0, 1])
    elements_rot = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.1, 'local_z': local_z_rot}]
    T = np.zeros((12, 12))
    for i in range(4):
        T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
    u_global_rot = T @ u_global
    K_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
    K_expected = T @ K_orig @ T.T
    assert np.allclose(K_rot, K_expected, atol=1e-10)