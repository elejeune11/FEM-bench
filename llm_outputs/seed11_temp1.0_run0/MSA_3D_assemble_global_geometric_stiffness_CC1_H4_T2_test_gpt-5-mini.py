def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elem_template = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 2e-06, 'J': 5e-07, 'local_z': np.array([0.0, 0.0, 1.0], dtype=float)}
    elements = [dict(elem_template, **{'nodes': (0, 1)}), dict(elem_template, **{'nodes': (1, 2)})]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u_zero = np.zeros(dof, dtype=float)
    K_zero = fcn(node_coords, elements, u_zero)
    assert K_zero.shape == (dof, dof)
    assert np.allclose(K_zero, np.zeros_like(K_zero), atol=1e-12, rtol=0)
    rng = np.random.RandomState(12345)
    u1 = 0.0001 * rng.randn(dof)
    u2 = 0.0001 * rng.randn(dof)
    K1 = fcn(node_coords, elements, u1)
    assert np.allclose(K1, K1.T, atol=1e-08, rtol=1e-08)
    alpha = 2.5
    K_alpha = fcn(node_coords, elements, alpha * u1)
    assert np.allclose(K_alpha, alpha * K1, atol=1e-07, rtol=1e-06)
    K_u1 = K1
    K_u2 = fcn(node_coords, elements, u2)
    K_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_sum, K_u1 + K_u2, atol=1e-07, rtol=1e-06)
    elements_rev = list(reversed(elements))
    K_rev = fcn(node_coords, elements_rev, u1)
    assert np.allclose(K_rev, K1, atol=1e-10, rtol=1e-10)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz]."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.5, 0.8, 0.3]], dtype=float)
    elem_template = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.02, 'I_y': 1.5e-06, 'I_z': 2.1e-06, 'J': 4e-07, 'local_z': np.array([0.0, 0.0, 1.0], dtype=float)}
    elements = [dict(elem_template, **{'nodes': (0, 1)}), dict(elem_template, **{'nodes': (1, 2)})]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    rng = np.random.RandomState(98765)
    u = 0.0001 * rng.randn(dof)
    K = fcn(node_coords, elements, u)
    assert K.shape == (dof, dof)
    theta = 0.37
    c = np.cos(theta)
    s = np.sin(theta)
    axis = np.array([1.0, 1.0, 0.0], dtype=float)
    axis = axis / np.linalg.norm(axis)
    (ux, uy, uz) = axis
    R = np.array([[c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s], [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s], [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)]], dtype=float)
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    for el in elements:
        el_rot = el.copy()
        if 'local_z' in el and el['local_z'] is not None:
            el_rot['local_z'] = R @ np.asarray(el['local_z'], dtype=float)
        elements_rot.append(el_rot)
    u_rot = np.zeros_like(u)
    for i in range(n_nodes):
        ut = u[6 * i:6 * i + 3]
        rt = u[6 * i + 3:6 * i + 6]
        u_rot[6 * i:6 * i + 3] = R @ ut
        u_rot[6 * i + 3:6 * i + 6] = R @ rt
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    T = np.zeros((dof, dof), dtype=float)
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    K_transformed = T @ K @ T.T
    assert np.allclose(K_rot, K_transformed, atol=1e-06, rtol=1e-06)