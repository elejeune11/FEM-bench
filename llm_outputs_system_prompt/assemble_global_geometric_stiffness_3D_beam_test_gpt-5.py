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
    np = __import__('numpy')
    mod = __import__(fcn.__module__)
    Q = np.eye(12, dtype=float)
    alpha = 1.0

    def beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        return np.eye(12, dtype=float)

    def compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        u_i = u_e_global[:6]
        u_j = u_e_global[6:]
        ti = u_i[:3]
        tj = u_j[:3]
        d = np.array([xj - xi, yj - yi, zj - zi], dtype=float)
        L = float(np.linalg.norm(d))
        if L == 0.0:
            d_hat = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            d_hat = d / L
        s = alpha * L * float(d_hat.dot(tj - ti))
        return np.full(12, s, dtype=float)

    def local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        s_eff = Fx2 + Mx2 + My1 + Mz1 + My2 + Mz2
        return s_eff * Q
    setattr(mod, 'beam_transformation_matrix_3D', beam_transformation_matrix_3D)
    setattr(mod, 'compute_local_element_loads_beam_3D', compute_local_element_loads_beam_3D)
    setattr(mod, 'local_geometric_stiffness_matrix_3D_beam', local_geometric_stiffness_matrix_3D_beam)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, -0.1], [2.0, -0.1, 0.3]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0]}]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u0 = np.zeros(dof, dtype=float)
    K0 = fcn(node_coords, elements, u0)
    assert np.allclose(K0, np.zeros_like(K0)), 'Zero displacement should produce zero geometric stiffness'
    u1 = np.arange(dof, dtype=float) / 10.0
    K1 = fcn(node_coords, elements, u1)
    assert np.allclose(K1, K1.T, atol=1e-12), 'Assembled geometric stiffness must be symmetric'
    c = 3.7
    K1_scaled = fcn(node_coords, elements, c * u1)
    assert np.allclose(K1_scaled, c * K1, atol=1e-12), 'Geometric stiffness should scale linearly with displacement scaling'
    u2 = (np.arange(dof, dtype=float)[::-1] - dof / 3.0) / 7.0
    K2 = fcn(node_coords, elements, u2)
    K12 = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K12, K1 + K2, atol=1e-12), 'Superposition must hold for independent displacement states'
    elements_reordered = [elements[1], elements[0]]
    K1_reordered = fcn(node_coords, elements_reordered, u1)
    assert np.allclose(K1_reordered, K1, atol=1e-12), 'Element order should not affect the assembled result'

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    np = __import__('numpy')
    mod = __import__(fcn.__module__)
    Q = np.eye(12, dtype=float)
    alpha = 1.0

    def beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        return np.eye(12, dtype=float)

    def compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        u_i = u_e_global[:6]
        u_j = u_e_global[6:]
        ti = u_i[:3]
        tj = u_j[:3]
        d = np.array([xj - xi, yj - yi, zj - zi], dtype=float)
        L = float(np.linalg.norm(d))
        if L == 0.0:
            d_hat = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            d_hat = d / L
        s = alpha * L * float(d_hat.dot(tj - ti))
        return np.full(12, s, dtype=float)

    def local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        s_eff = Fx2 + Mx2 + My1 + Mz1 + My2 + Mz2
        return s_eff * Q
    setattr(mod, 'beam_transformation_matrix_3D', beam_transformation_matrix_3D)
    setattr(mod, 'compute_local_element_loads_beam_3D', compute_local_element_loads_beam_3D)
    setattr(mod, 'local_geometric_stiffness_matrix_3D_beam', local_geometric_stiffness_matrix_3D_beam)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.2], [2.0, -0.3, 0.7]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 2.0, 'I_rho': 3.0, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 1.5, 'I_rho': 2.5, 'local_z': [0.0, 0.0, 1.0]}]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u = np.arange(dof, dtype=float) / 5.0 - 0.3
    axis = np.array([1.0, 2.0, 3.0], dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = 0.613
    Kmat = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]], dtype=float)
    R = np.eye(3) + np.sin(theta) * Kmat + (1.0 - np.cos(theta)) * (Kmat @ Kmat)
    T = np.zeros((dof, dof), dtype=float)
    for a in range(n_nodes):
        i = 6 * a
        T[i:i + 3, i:i + 3] = R
        T[i + 3:i + 6, i + 3:i + 6] = R
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for ele in elements:
        lz = np.array(ele.get('local_z', [0.0, 0.0, 1.0]), dtype=float)
        lz_rot = (R @ lz).tolist()
        ele_rot = dict(ele)
        ele_rot['local_z'] = lz_rot
        elements_rot.append(ele_rot)
    K = fcn(node_coords, elements, u)
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    TKTT = T @ K @ T.T
    assert np.allclose(K_rot, TKTT, atol=1e-12), 'Geometric stiffness must transform objectively under global rotations'