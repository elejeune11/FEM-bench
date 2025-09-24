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
    globs = fcn.__globals__

    def beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        pi = np.array([xi, yi, zi], dtype=float)
        pj = np.array([xj, yj, zj], dtype=float)
        ex = pj - pi
        L = np.linalg.norm(ex)
        if L == 0:
            raise ValueError('Zero-length element')
        ex /= L
        if local_z is None:
            ez_dir = np.array([0.0, 0.0, 1.0])
        else:
            ez_dir = np.array(local_z, dtype=float)
            if np.linalg.norm(ez_dir) == 0:
                ez_dir = np.array([0.0, 0.0, 1.0])
        ez_dir /= np.linalg.norm(ez_dir)
        if abs(np.dot(ex, ez_dir)) > 0.999:
            ez_dir = np.array([0.0, 1.0, 0.0])
        ey = np.cross(ez_dir, ex)
        ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        R = np.stack([ex, ey, ez], axis=1)
        Tn = np.zeros((6, 6))
        Tn[:3, :3] = R
        Tn[3:, 3:] = R
        Gamma = np.zeros((12, 12))
        Gamma[:6, :6] = Tn
        Gamma[6:, 6:] = Tn
        return Gamma

    def compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        Gamma = globs['beam_transformation_matrix_3D'](xi, yi, zi, xj, yj, zj, ele.get('local_z', None))
        d_l = Gamma @ u_e_global
        w = np.arange(1, 13, dtype=float)
        D = np.diag(w)
        f_l = D @ d_l
        return f_l

    def local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        s = 1.0 * Fx2 + 2.0 * Mx2 + 3.0 * My1 + 4.0 * Mz1 + 5.0 * My2 + 6.0 * Mz2
        K = s * np.eye(12)
        return K
    globs['beam_transformation_matrix_3D'] = beam_transformation_matrix_3D
    globs['compute_local_element_loads_beam_3D'] = compute_local_element_loads_beam_3D
    globs['local_geometric_stiffness_matrix_3D_beam'] = local_geometric_stiffness_matrix_3D_beam
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0]}]
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    u_zero = np.zeros(ndof)
    K_zero = fcn(node_coords, elements, u_zero)
    assert K_zero.shape == (ndof, ndof)
    assert np.allclose(K_zero, 0.0)
    u1 = np.arange(1, ndof + 1, dtype=float) * 0.01
    K1 = fcn(node_coords, elements, u1)
    assert np.allclose(K1, K1.T, atol=1e-12, rtol=0.0)
    scale = 3.5
    K_scaled = fcn(node_coords, elements, scale * u1)
    assert np.allclose(K_scaled, scale * K1, atol=1e-12, rtol=1e-12)
    u2 = np.arange(1, ndof + 1, dtype=float)[::-1] * -0.02
    K_u1 = K1
    K_u2 = fcn(node_coords, elements, u2)
    K_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_sum, K_u1 + K_u2, atol=1e-12, rtol=1e-12)
    elements_reversed = list(reversed(elements))
    K1_reordered = fcn(node_coords, elements_reversed, u1)
    assert np.allclose(K1_reordered, K1, atol=1e-12, rtol=0.0)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity: rotating the entire system (geometry, local axes,
    and displacement field) by a global rotation R produces a geometric stiffness
    K_g^rot that satisfies K_g^rot ≈ T K_g T^T, where T is block-diagonal with
    per-node blocks diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    globs = fcn.__globals__

    def beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        pi = np.array([xi, yi, zi], dtype=float)
        pj = np.array([xj, yj, zj], dtype=float)
        ex = pj - pi
        L = np.linalg.norm(ex)
        if L == 0:
            raise ValueError('Zero-length element')
        ex /= L
        if local_z is None:
            ez_dir = np.array([0.0, 0.0, 1.0])
        else:
            ez_dir = np.array(local_z, dtype=float)
            if np.linalg.norm(ez_dir) == 0:
                ez_dir = np.array([0.0, 0.0, 1.0])
        ez_dir /= np.linalg.norm(ez_dir)
        if abs(np.dot(ex, ez_dir)) > 0.999:
            ez_dir = np.array([0.0, 1.0, 0.0])
        ey = np.cross(ez_dir, ex)
        ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        R = np.stack([ex, ey, ez], axis=1)
        Tn = np.zeros((6, 6))
        Tn[:3, :3] = R
        Tn[3:, 3:] = R
        Gamma = np.zeros((12, 12))
        Gamma[:6, :6] = Tn
        Gamma[6:, 6:] = Tn
        return Gamma

    def compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        Gamma = globs['beam_transformation_matrix_3D'](xi, yi, zi, xj, yj, zj, ele.get('local_z', None))
        d_l = Gamma @ u_e_global
        w = np.arange(1, 13, dtype=float)
        D = np.diag(w)
        f_l = D @ d_l
        return f_l

    def local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        s = 1.0 * Fx2 + 2.0 * Mx2 + 3.0 * My1 + 4.0 * Mz1 + 5.0 * My2 + 6.0 * Mz2
        K = s * np.eye(12)
        return K
    globs['beam_transformation_matrix_3D'] = beam_transformation_matrix_3D
    globs['compute_local_element_loads_beam_3D'] = compute_local_element_loads_beam_3D
    globs['local_geometric_stiffness_matrix_3D_beam'] = local_geometric_stiffness_matrix_3D_beam
    node_coords = np.array([[0.2, -0.1, 0.3], [1.0, 0.5, -0.2], [2.2, 1.0, 0.4]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.2, 'I_rho': 0.8, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 0.9, 'I_rho': 1.1, 'local_z': [0.0, 0.0, 1.0]}]
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    u = np.linspace(-0.05, 0.07, ndof)
    K = fcn(node_coords, elements, u)
    axis = np.array([0.3, 0.5, 0.8], dtype=float)
    axis /= np.linalg.norm(axis)
    theta = 0.7
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=float)
    R = np.eye(3) + np.sin(theta) * Kx + (1 - np.cos(theta)) * (Kx @ Kx)
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for ele in elements:
        lz = np.array(ele['local_z'], dtype=float)
        lz_rot = (R @ lz).tolist()
        e2 = dict(ele)
        e2['local_z'] = lz_rot
        elements_rot.append(e2)
    T = np.zeros((ndof, ndof))
    for i in range(n_nodes):
        Ti = np.zeros((6, 6))
        Ti[:3, :3] = R
        Ti[3:, 3:] = R
        T[6 * i:6 * (i + 1), 6 * i:6 * (i + 1)] = Ti
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    TKTT = T @ K @ T.T
    assert np.allclose(K_rot, TKTT, atol=1e-12, rtol=1e-12)