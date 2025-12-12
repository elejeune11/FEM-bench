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

    def beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        pi = np.array([xi, yi, zi], dtype=float)
        pj = np.array([xj, yj, zj], dtype=float)
        vx = pj - pi
        L = np.linalg.norm(vx)
        if L == 0:
            raise ValueError('Zero-length element.')
        ex = vx / L
        if local_z is None:
            ez_ref = np.array([0.0, 0.0, 1.0])
        else:
            ez_ref = np.array(local_z, dtype=float)
        ez_temp = ez_ref - np.dot(ez_ref, ex) * ex
        if np.linalg.norm(ez_temp) < 1e-12:
            ez_ref = np.array([0.0, 1.0, 0.0])
            ez_temp = ez_ref - np.dot(ez_ref, ex) * ex
        ez = ez_temp / np.linalg.norm(ez_temp)
        ey = np.cross(ez, ex)
        ey = ey / np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        R_lg = np.column_stack((ex, ey, ez))
        R_gl = R_lg.T
        Gamma = np.zeros((12, 12), dtype=float)
        for k in range(4):
            Gamma[3 * k:3 * k + 3, 3 * k:3 * k + 3] = R_gl
        return Gamma

    def compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele.get('local_z'))
        d_l = Gamma @ u_e_global
        weights = np.linspace(1.0, 2.2, 12)
        S = np.diag(weights)
        loads_local = S @ d_l
        return loads_local

    def local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        scale = 1.0 / max(L, 1e-12)
        base = np.arange(1, 13, dtype=float)
        W0 = np.diag(base * 1.0 * scale)
        W1 = np.diag(base * 0.8 * scale)
        W2 = np.diag(base * 0.6 * scale)
        W3 = np.diag(base * 0.4 * scale)
        W4 = np.diag(base * 0.5 * scale)
        W5 = np.diag(base * 0.3 * scale)
        K = Fx2 * W0 + Mx2 * W1 + My1 * W2 + Mz1 * W3 + My2 * W4 + Mz2 * W5
        return K
    fcn.__globals__['beam_transformation_matrix_3D'] = beam_transformation_matrix_3D
    fcn.__globals__['compute_local_element_loads_beam_3D'] = compute_local_element_loads_beam_3D
    fcn.__globals__['local_geometric_stiffness_matrix_3D_beam'] = local_geometric_stiffness_matrix_3D_beam
    L = 2.0
    node_coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [2 * L, 0.0, 0.0]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.5, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 1.2, 'I_rho': 0.4, 'local_z': [0.0, 0.0, 1.0]}]
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    u_zero = np.zeros(ndof, dtype=float)
    K_zero = fcn(node_coords, elements, u_zero)
    assert K_zero.shape == (ndof, ndof)
    assert np.allclose(K_zero, np.zeros((ndof, ndof)), atol=1e-12)
    u1 = np.zeros(ndof, dtype=float)
    for n in range(n_nodes):
        for d in range(6):
            u1[6 * n + d] = 0.1 * (n + 1) * (d + 1)
    u2 = np.zeros(ndof, dtype=float)
    for n in range(n_nodes):
        for d in range(6):
            u2[6 * n + d] = 0.05 * (n + 2) * (6 - d)
    K1 = fcn(node_coords, elements, u1)
    assert np.allclose(K1, K1.T, atol=1e-10)
    alpha = 3.0
    K_alpha = fcn(node_coords, elements, alpha * u1)
    assert np.allclose(K_alpha, alpha * K1, atol=1e-10)
    K2 = fcn(node_coords, elements, u2)
    K12 = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K12, K1 + K2, atol=1e-10)
    elements_reversed = list(reversed(elements))
    K1_rev = fcn(node_coords, elements_reversed, u1)
    assert np.allclose(K1_rev, K1, atol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """

    def beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        pi = np.array([xi, yi, zi], dtype=float)
        pj = np.array([xj, yj, zj], dtype=float)
        vx = pj - pi
        L = np.linalg.norm(vx)
        if L == 0:
            raise ValueError('Zero-length element.')
        ex = vx / L
        if local_z is None:
            ez_ref = np.array([0.0, 0.0, 1.0])
        else:
            ez_ref = np.array(local_z, dtype=float)
        ez_temp = ez_ref - np.dot(ez_ref, ex) * ex
        if np.linalg.norm(ez_temp) < 1e-12:
            ez_ref = np.array([0.0, 1.0, 0.0])
            ez_temp = ez_ref - np.dot(ez_ref, ex) * ex
        ez = ez_temp / np.linalg.norm(ez_temp)
        ey = np.cross(ez, ex)
        ey = ey / np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        R_lg = np.column_stack((ex, ey, ez))
        R_gl = R_lg.T
        Gamma = np.zeros((12, 12), dtype=float)
        for k in range(4):
            Gamma[3 * k:3 * k + 3, 3 * k:3 * k + 3] = R_gl
        return Gamma

    def compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele.get('local_z'))
        d_l = Gamma @ u_e_global
        weights = np.linspace(1.0, 2.2, 12)
        S = np.diag(weights)
        loads_local = S @ d_l
        return loads_local

    def local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        scale = 1.0 / max(L, 1e-12)
        base = np.arange(1, 13, dtype=float)
        W0 = np.diag(base * 1.0 * scale)
        W1 = np.diag(base * 0.8 * scale)
        W2 = np.diag(base * 0.6 * scale)
        W3 = np.diag(base * 0.4 * scale)
        W4 = np.diag(base * 0.5 * scale)
        W5 = np.diag(base * 0.3 * scale)
        K = Fx2 * W0 + Mx2 * W1 + My1 * W2 + Mz1 * W3 + My2 * W4 + Mz2 * W5
        return K
    fcn.__globals__['beam_transformation_matrix_3D'] = beam_transformation_matrix_3D
    fcn.__globals__['compute_local_element_loads_beam_3D'] = compute_local_element_loads_beam_3D
    fcn.__globals__['local_geometric_stiffness_matrix_3D_beam'] = local_geometric_stiffness_matrix_3D_beam
    L = 2.0
    node_coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [2 * L, 0.0, 0.0]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.5, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 1.2, 'I_rho': 0.4, 'local_z': [0.0, 0.0, 1.0]}]
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    u = np.zeros(ndof, dtype=float)
    for n in range(n_nodes):
        for d in range(6):
            u[6 * n + d] = 0.15 * (n + 1) * (d + 2)

    def Rz(angle):
        (c, s) = (np.cos(angle), np.sin(angle))
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    def Ry(angle):
        (c, s) = (np.cos(angle), np.sin(angle))
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
    R = Rz(0.7) @ Ry(0.3)
    K = fcn(node_coords, elements, u)
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for ele in elements:
        z0 = np.array(ele['local_z'], dtype=float)
        z_rot = z0 @ R.T
        elements_rot.append({'node_i': ele['node_i'], 'node_j': ele['node_j'], 'A': ele['A'], 'I_rho': ele['I_rho'], 'local_z': z_rot.tolist()})
    T_block = np.zeros((6, 6), dtype=float)
    T_block[0:3, 0:3] = R
    T_block[3:6, 3:6] = R
    T = np.kron(np.eye(n_nodes), T_block)
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    assert np.allclose(K_rot, T @ K @ T.T, atol=1e-10)