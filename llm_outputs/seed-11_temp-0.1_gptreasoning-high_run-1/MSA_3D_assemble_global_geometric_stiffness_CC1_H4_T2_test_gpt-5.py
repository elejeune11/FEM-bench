def test_multi_element_core_correctness_assembly(fcn):
    """
    Verify basic correctness for a simple 3-node, 2-element chain:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.3, 0.2, 0.4], [2.8, -0.1, 0.9]], dtype=float)
    common_props = dict(E=210000000000.0, nu=0.3, A=0.0005, I_y=8e-08, I_z=7e-08, J=1.5e-07)
    elements = [dict(node_i=0, node_j=1, local_z=np.array([0.0, 0.0, 1.0], dtype=float), **common_props), dict(node_i=1, node_j=2, local_z=np.array([0.0, 0.0, 1.0], dtype=float), **common_props)]
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    u_zero = np.zeros(ndof, dtype=float)
    K0 = fcn(node_coords, elements, u_zero)
    assert K0.shape == (ndof, ndof)
    assert np.allclose(K0, np.zeros_like(K0), atol=1e-12)
    u1 = np.array([0.0001, -0.0002, 5e-05, 2e-05, -3e-05, 4e-05, -8e-05, 0.00011, -6e-05, -3.5e-05, 2.5e-05, -1.5e-05, 9e-05, -4e-05, 3e-05, -2e-05, 1e-05, 7e-06], dtype=float)
    K1 = fcn(node_coords, elements, u1)
    assert np.allclose(K1, K1.T, rtol=1e-09, atol=1e-09)
    alpha = 3.2
    K_alpha = fcn(node_coords, elements, alpha * u1)
    assert np.allclose(K_alpha, alpha * K1, rtol=1e-08, atol=1e-12)
    u2 = np.array([-3e-05, 4e-05, -2.5e-05, 6e-06, 4e-06, -8e-06, 7e-05, -9e-05, 4e-05, -1.2e-05, 1.8e-05, 2.1e-05, -4e-05, 2.2e-05, -1.1e-05, 3.3e-05, -2.7e-05, 1.4e-05], dtype=float)
    K2 = fcn(node_coords, elements, u2)
    K12 = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K12, K1 + K2, rtol=1e-08, atol=1e-12)
    elements_reversed = [dict(node_i=1, node_j=2, local_z=np.array([0.0, 0.0, 1.0], dtype=float), **common_props), dict(node_i=0, node_j=1, local_z=np.array([0.0, 0.0, 1.0], dtype=float), **common_props)]
    K1_rev = fcn(node_coords, elements_reversed, u1)
    assert np.allclose(K1_rev, K1, rtol=1e-09, atol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity: Rotating the entire system (geometry, local axes,
    and displacement field) by a global rotation R produces K_g^rot ≈ T K_g T^T,
    where T is block-diagonal with per-node blocks diag(R, R).
    """

    def rot_axis_angle(axis, angle):
        a = np.asarray(axis, dtype=float)
        a = a / np.linalg.norm(a)
        ax = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]], dtype=float)
        I = np.eye(3)
        return I * np.cos(angle) + (1 - np.cos(angle)) * np.outer(a, a) + np.sin(angle) * ax

    def build_T(n_nodes, R):
        T = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
        for i in range(n_nodes):
            i6 = 6 * i
            T[i6:i6 + 3, i6:i6 + 3] = R
            T[i6 + 3:i6 + 6, i6 + 3:i6 + 6] = R
        return T
    node_coords = np.array([[0.0, 0.0, 0.0], [1.1, 0.4, 0.3], [2.2, -0.2, 1.0]], dtype=float)
    common_props = dict(E=200000000000.0, nu=0.29, A=0.0006, I_y=9e-08, I_z=6e-08, J=1.2e-07)
    local_z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
    elements = [dict(node_i=0, node_j=1, local_z=local_z_ref.copy(), **common_props), dict(node_i=1, node_j=2, local_z=local_z_ref.copy(), **common_props)]
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    u = np.array([7e-05, -4e-05, 2e-05, 1e-05, -2e-05, 3e-05, -2.5e-05, 3.5e-05, -1.5e-05, 2.2e-05, 1.7e-05, -2.1e-05, 4.5e-05, -1.2e-05, 8e-06, -7e-06, 5e-06, 9e-06], dtype=float)
    K = fcn(node_coords, elements, u)
    assert K.shape == (ndof, ndof)
    axis = np.array([0.3, 0.5, 0.8], dtype=float)
    angle = 0.7
    R = rot_axis_angle(axis, angle)
    T = build_T(n_nodes, R)
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for e in elements:
        e_rot = e.copy()
        e_rot['local_z'] = R @ np.asarray(e['local_z'], dtype=float)
        elements_rot.append(e_rot)
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_expected = T @ K @ T.T
    assert np.allclose(K_rot, K_expected, rtol=5e-08, atol=1e-10)