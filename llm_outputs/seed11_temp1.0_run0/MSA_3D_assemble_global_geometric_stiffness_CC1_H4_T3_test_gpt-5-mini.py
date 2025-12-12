def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elem_props = dict(E=210000000000.0, nu=0.3, A=0.001, I_y=1e-06, I_z=1e-06, J=2e-06, local_z=np.array([0.0, 0.0, 1.0]))
    elements = [{**elem_props, 'nodes': (0, 1)}, {**elem_props, 'nodes': (1, 2)}]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u_zero = np.zeros(dof)
    K_zero = fcn(node_coords, elements, u_zero)
    assert isinstance(K_zero, np.ndarray)
    assert K_zero.shape == (dof, dof)
    assert np.allclose(K_zero, np.zeros((dof, dof)), atol=1e-12, rtol=0.0)
    u1 = np.zeros(dof)
    u2 = np.zeros(dof)
    u1[1] = 0.0001
    u1[7] = -0.0002
    u2[2] = 5e-05
    u2[8] = 0.0001
    K_u1 = fcn(node_coords, elements, u1)
    K_u2 = fcn(node_coords, elements, u2)
    K_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_sum, K_sum.T, rtol=1e-06, atol=1e-09)
    alpha = 3.7
    K_alpha = fcn(node_coords, elements, alpha * u1)
    assert np.allclose(K_alpha, alpha * K_u1, rtol=1e-06, atol=1e-09)
    assert np.allclose(K_sum, K_u1 + K_u2, rtol=1e-06, atol=1e-09)
    elements_reversed = list(reversed(elements))
    K_reversed = fcn(node_coords, elements_reversed, u1 + u2)
    assert np.allclose(K_sum, K_reversed, rtol=1e-09, atol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, 0.1, 0.0]])
    elem_props = dict(E=210000000000.0, nu=0.3, A=0.002, I_y=2e-06, I_z=1.5e-06, J=3e-06, local_z=np.array([0.0, 0.0, 1.0]))
    elements = [{**elem_props, 'nodes': (0, 1)}, {**elem_props, 'nodes': (1, 2)}]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u = np.zeros(dof)
    rng = np.random.default_rng(42)
    u += rng.standard_normal(dof) * 0.0001
    K_orig = fcn(node_coords, elements, u)
    assert K_orig.shape == (dof, dof)
    axis = np.array([1.0, 1.0, 0.3])
    axis = axis / np.linalg.norm(axis)
    theta = 0.4
    K_axis = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta) * K_axis + (1 - np.cos(theta)) * (K_axis @ K_axis)
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    for el in elements:
        el_rot = el.copy()
        el_rot['local_z'] = (R @ np.asarray(el['local_z'])).reshape(3)
        el_rot['nodes'] = el['nodes']
        elements_rot.append(el_rot)
    T = np.zeros((dof, dof))
    for i in range(n_nodes):
        block = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
        T[6 * i:6 * i + 6, 6 * i:6 * i + 6] = block
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_transformed = T @ K_orig @ T.T
    assert np.allclose(K_rot, K_transformed, rtol=1e-06, atol=1e-08)