def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    import numpy as np
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0], 'E': 1.0}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0], 'E': 1.0}]
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    u_zero = np.zeros(n_dof, dtype=float)
    K_zero = np.asarray(fcn(node_coords, elements, u_zero))
    assert K_zero.shape == (n_dof, n_dof)
    assert np.allclose(K_zero, 0.0, atol=1e-10, rtol=1e-12)
    u_base = np.zeros(n_dof, dtype=float)
    du = 0.001
    u_base[0] = -du
    u_base[6 * 2 + 0] = du
    K_base = np.asarray(fcn(node_coords, elements, u_base))
    assert K_base.shape == (n_dof, n_dof)
    assert np.allclose(K_base, K_base.T, atol=1e-08, rtol=1e-10)
    alpha = 2.5
    K_scaled = np.asarray(fcn(node_coords, elements, alpha * u_base))
    assert np.allclose(K_scaled, alpha * K_base, atol=1e-08, rtol=1e-06)
    u_add = np.zeros(n_dof, dtype=float)
    u_add[1 + 6 * 1] = 0.002
    K_add = np.asarray(fcn(node_coords, elements, u_add))
    K_combined = np.asarray(fcn(node_coords, elements, u_base + u_add))
    assert np.allclose(K_combined, K_base + K_add, atol=1e-08, rtol=1e-06)
    elements_reordered = list(reversed(elements))
    K_reordered = np.asarray(fcn(node_coords, elements_reordered, u_base))
    assert np.allclose(K_reordered, K_base, atol=1e-10, rtol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    import numpy as np
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0], 'E': 1.0}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0], 'E': 1.0}]
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    u = np.zeros(n_dof, dtype=float)
    for i in range(n_nodes):
        base = 6 * i
        u[base:base + 3] = np.array([0.001 * (i - 1), 0.002 * (i - 1), 0.0005 * (i - 1)])
        u[base + 3:base + 6] = np.array([0.0001 * (i + 1), -5e-05 * (i + 1), 0.0002 * (i + 1)])
    K_orig = np.asarray(fcn(node_coords, elements, u))
    theta = 0.37
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for ele in elements:
        ele_rot = dict(ele)
        if 'local_z' in ele_rot and ele_rot['local_z'] is not None:
            lz = np.asarray(ele_rot['local_z'], dtype=float)
            ele_rot['local_z'] = (R @ lz).tolist()
        elements_rot.append(ele_rot)
    u_rot = np.zeros_like(u)
    for i in range(n_nodes):
        base = 6 * i
        trans = u[base:base + 3]
        rot = u[base + 3:base + 6]
        u_rot[base:base + 3] = R @ trans
        u_rot[base + 3:base + 6] = R @ rot
    K_rot = np.asarray(fcn(node_coords_rot, elements_rot, u_rot))
    block = np.zeros((6, 6), dtype=float)
    block[0:3, 0:3] = R
    block[3:6, 3:6] = R
    T = np.kron(np.eye(n_nodes, dtype=float), block)
    K_transformed = T @ K_orig @ T.T
    assert K_rot.shape == K_transformed.shape == (n_dof, n_dof)
    assert np.allclose(K_rot, K_transformed, atol=1e-06, rtol=1e-06)