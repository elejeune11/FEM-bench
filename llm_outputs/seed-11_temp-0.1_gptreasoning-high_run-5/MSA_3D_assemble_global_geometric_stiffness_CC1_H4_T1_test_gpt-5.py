def test_multi_element_core_correctness_assembly(fcn):
    """
    Verify basic correctness for a simple 3-node, 2-element chain.
    Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    import numpy as np
    node_coords = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.6, 0.0, 0.0]], dtype=float)
    A = 0.01
    I_rho = 0.0001
    local_z = [0.0, 0.0, 1.0]
    elements = [{'node_i': 0, 'node_j': 1, 'A': A, 'I_rho': I_rho, 'local_z': local_z}, {'node_i': 1, 'node_j': 2, 'A': A, 'I_rho': I_rho, 'local_z': local_z}]
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    u0 = np.zeros(ndof, dtype=float)
    K0 = fcn(node_coords, elements, u0)
    assert K0.shape == (ndof, ndof)
    assert np.allclose(K0, 0.0, atol=1e-12)
    rng = np.random.default_rng(12345)
    u1 = rng.standard_normal(ndof) * 0.001
    K1 = fcn(node_coords, elements, u1)
    assert np.allclose(K1, K1.T, rtol=1e-09, atol=1e-11)
    alpha = 2.5
    K_alpha = fcn(node_coords, elements, alpha * u1)
    assert np.allclose(K_alpha, alpha * K1, rtol=1e-08, atol=1e-12)
    u2 = rng.standard_normal(ndof) * 0.001
    K2 = fcn(node_coords, elements, u2)
    K12 = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K12, K1 + K2, rtol=1e-08, atol=1e-11)
    elements_reversed = list(reversed(elements))
    K1_rev = fcn(node_coords, elements_reversed, u1)
    assert np.allclose(K1_rev, K1, rtol=1e-12, atol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity: Rotating the entire system (geometry, local axes,
    and displacement field) by a global rotation R should produce
    K_g^rot ≈ T K_g T^T, where T has per-node blocks diag(R, R).
    """
    import numpy as np

    def Rx(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=float)

    def Ry(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=float)

    def Rz(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=float)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.3, 0.2, -0.1], [2.8, 0.4, 0.7]], dtype=float)
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    A = 0.012
    I_rho = 0.0002
    local_z = np.array([0.0, 0.0, 1.0], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': A, 'I_rho': I_rho, 'local_z': local_z.tolist()}, {'node_i': 1, 'node_j': 2, 'A': A, 'I_rho': I_rho, 'local_z': local_z.tolist()}]
    rng = np.random.default_rng(9876)
    u = rng.standard_normal(ndof) * 0.0001
    K = fcn(node_coords, elements, u)
    ax, ay, az = np.deg2rad([20.0, -15.0, 30.0])
    R = Rz(az) @ Ry(ay) @ Rx(ax)
    T_node = np.zeros((6, 6), dtype=float)
    T_node[:3, :3] = R
    T_node[3:, 3:] = R
    T = np.kron(np.eye(n_nodes), T_node)
    node_coords_rot = node_coords @ R.T
    u_rot = T @ u
    elements_rot = []
    for e in elements:
        z_rot = (R @ np.array(e['local_z'], dtype=float)).tolist()
        e_rot = dict(e)
        e_rot['local_z'] = z_rot
        elements_rot.append(e_rot)
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_expected = T @ K @ T.T
    assert np.allclose(K_rot, K_expected, rtol=1e-08, atol=1e-10)