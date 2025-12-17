def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result."""
    L = 2.5
    node_coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [2 * L, 0.0, 0.0]], dtype=float)
    E = 210000000000.0
    nu = 0.3
    A = 0.008
    I_y = 6e-06
    I_z = 6e-06
    J = 1.2e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u_zero = np.zeros(dof)
    K0 = fcn(node_coords, elements, u_zero)
    assert K0.shape == (dof, dof)
    assert np.allclose(K0, 0.0, atol=1e-12)
    u1 = np.zeros(dof)
    delta = 0.001
    u1[0] = 0.0
    u1[6 + 0] = delta
    u1[12 + 0] = 2 * delta
    K1 = fcn(node_coords, elements, u1)
    assert np.allclose(K1, K1.T, atol=1e-10, rtol=1e-10)
    alpha = 3.5
    K1_scaled = fcn(node_coords, elements, alpha * u1)
    assert np.allclose(K1_scaled, alpha * K1, atol=1e-09, rtol=1e-09)
    u2 = np.zeros(dof)
    beta = -0.0004
    u2[6 + 1] = beta
    Ku2 = fcn(node_coords, elements, u2)
    u12 = u1 + u2
    Ku12 = fcn(node_coords, elements, u12)
    assert np.allclose(Ku12, K1 + Ku2, atol=1e-09, rtol=1e-09)
    elements_reordered = [elements[1], elements[0]]
    Ku12_reordered = fcn(node_coords, elements_reordered, u12)
    assert np.allclose(Ku12_reordered, Ku12, atol=1e-12, rtol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce K_g^rot ≈ T K_g T^T, where T is block-diagonal
    with per-node blocks diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz]."""
    L = 1.7
    node_coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [2 * L, 0.0, 0.0]], dtype=float)
    E = 200000000000.0
    nu = 0.29
    A = 0.005
    I_y = 4e-06
    I_z = 6e-06
    J = 9e-06
    base_elem = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    elements = [dict(base_elem, **{'node_i': 0, 'node_j': 1}), dict(base_elem, **{'node_i': 1, 'node_j': 2})]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u = np.zeros(dof)
    ax = 0.0008
    by = 0.0003
    cz = -0.0002
    u[0] = 0.0
    u[6 + 0] = ax
    u[12 + 0] = 2 * ax
    u[6 + 1] = by
    u[12 + 2] = cz
    K = fcn(node_coords, elements, u)
    phi = 0.7
    c = np.cos(phi)
    s = np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for e in elements:
        z_ref = np.asarray(e['local_z'], dtype=float)
        z_rot = z_ref @ R.T
        e_rot = dict(e)
        e_rot['local_z'] = z_rot
        elements_rot.append(e_rot)
    u_rot = np.zeros_like(u)
    for i in range(n_nodes):
        ui = u[6 * i:6 * i + 3]
        ri = u[6 * i + 3:6 * i + 6]
        u_rot[6 * i:6 * i + 3] = R @ ui
        u_rot[6 * i + 3:6 * i + 6] = R @ ri
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    T = np.zeros((dof, dof))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    K_pred = T @ K @ T.T
    assert np.allclose(K_rot, K_pred, atol=5e-09, rtol=5e-09)