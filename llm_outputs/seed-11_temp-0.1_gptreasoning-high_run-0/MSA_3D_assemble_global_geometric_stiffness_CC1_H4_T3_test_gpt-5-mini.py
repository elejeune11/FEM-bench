def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    n_nodes = 3
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-06
    I_z = 1e-06
    J = 1e-06
    elem0 = {'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    elem1 = {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    elements = [elem0, elem1]
    dof = 6 * n_nodes
    u_zero = np.zeros(dof)
    K_zero = fcn(node_coords, elements, u_zero)
    assert K_zero.shape == (dof, dof)
    assert np.allclose(K_zero, np.zeros_like(K_zero), atol=1e-12, rtol=0.0)
    rng = np.random.RandomState(42)
    u = 1e-05 * (rng.rand(dof) - 0.5)
    K = fcn(node_coords, elements, u)
    assert K.shape == (dof, dof)
    assert np.allclose(K, K.T, atol=1e-08, rtol=1e-06)
    a = 2.5
    K_a = fcn(node_coords, elements, a * u)
    assert np.allclose(K_a, a * K, rtol=1e-06, atol=1e-10)
    u1 = 1e-05 * (rng.rand(dof) - 0.5)
    u2 = 1e-05 * (rng.rand(dof) - 0.5)
    K1 = fcn(node_coords, elements, u1)
    K2 = fcn(node_coords, elements, u2)
    K_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_sum, K1 + K2, rtol=1e-06, atol=1e-10)
    elements_reordered = [elem1, elem0]
    K_reordered = fcn(node_coords, elements_reordered, u)
    assert np.allclose(K_reordered, K, rtol=1e-12, atol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    n_nodes = 3
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-06
    I_z = 1e-06
    J = 1e-06
    local_z = np.array([0.0, 0.0, 1.0])
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z.copy()}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z.copy()}]
    dof = 6 * n_nodes
    rng = np.random.RandomState(7)
    u = 1e-05 * (rng.rand(dof) - 0.5)
    K = fcn(node_coords, elements, u)
    theta = 0.73
    axis = np.array([1.0, 1.0, 0.2])
    axis = axis / np.linalg.norm(axis)
    K_mat = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    R = np.eye(3) * np.cos(theta) + (1.0 - np.cos(theta)) * np.outer(axis, axis) + np.sin(theta) * K_mat
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    for el in elements:
        el_rot = el.copy()
        if el.get('local_z') is not None:
            el_rot['local_z'] = R @ np.asarray(el['local_z'])
        elements_rot.append(el_rot)
    u_rot = np.zeros_like(u)
    for i in range(n_nodes):
        ut = u[6 * i:6 * i + 3]
        rt = u[6 * i + 3:6 * i + 6]
        u_rot[6 * i:6 * i + 3] = R @ ut
        u_rot[6 * i + 3:6 * i + 6] = R @ rt
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    T = np.zeros((dof, dof))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    K_transformed = T @ K @ T.T
    assert K_rot.shape == K_transformed.shape
    assert np.allclose(K_rot, K_transformed, rtol=1e-06, atol=1e-09)