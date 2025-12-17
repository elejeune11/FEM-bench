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
        t = pj - pi
        L = np.linalg.norm(t)
        if L == 0.0:
            R = np.eye(3)
        else:
            x_hat = t / L
            if local_z is None:
                v = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                v = np.array(local_z, dtype=float)
            v = v / (np.linalg.norm(v) + 1e-15)
            z_temp = v - np.dot(v, x_hat) * x_hat
            nz = np.linalg.norm(z_temp)
            if nz < 1e-12:
                v = np.array([0.0, 1.0, 0.0], dtype=float)
                z_temp = v - np.dot(v, x_hat) * x_hat
                nz = np.linalg.norm(z_temp)
                if nz < 1e-12:
                    R = np.eye(3)
                else:
                    z_hat = z_temp / nz
                    y_hat = np.cross(z_hat, x_hat)
                    y_hat = y_hat / (np.linalg.norm(y_hat) + 1e-15)
                    z_hat = np.cross(x_hat, y_hat)
                    R = np.vstack([x_hat, y_hat, z_hat])
            else:
                z_hat = z_temp / nz
                y_hat = np.cross(z_hat, x_hat)
                y_hat = y_hat / (np.linalg.norm(y_hat) + 1e-15)
                z_hat = np.cross(x_hat, y_hat)
                R = np.vstack([x_hat, y_hat, z_hat])
        R6 = np.zeros((6, 6), dtype=float)
        R6[:3, :3] = R
        R6[3:, 3:] = R
        Gamma = np.zeros((12, 12), dtype=float)
        Gamma[:6, :6] = R6
        Gamma[6:, 6:] = R6
        return Gamma

    def compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele.get('local_z', None))
        d_local = Gamma @ u_e_global
        weights = np.arange(1, 13, dtype=float)
        return weights * d_local

    def local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        s = Fx2 + Mx2 + My1 + Mz1 + My2 + Mz2
        return np.eye(12) * s
    g = fcn.__globals__
    g['beam_transformation_matrix_3D'] = beam_transformation_matrix_3D
    g['compute_local_element_loads_beam_3D'] = compute_local_element_loads_beam_3D
    g['local_geometric_stiffness_matrix_3D_beam'] = local_geometric_stiffness_matrix_3D_beam
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0]}]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u_zero = np.zeros(dof, dtype=float)
    K0 = fcn(node_coords, elements, u_zero)
    assert K0.shape == (dof, dof)
    assert np.allclose(K0, 0.0)
    u_test = np.arange(dof, dtype=float) / 10.0 + 1.0
    K = fcn(node_coords, elements, u_test)
    assert np.allclose(K, K.T, atol=1e-12)
    scale = 3.2
    K_scaled = fcn(node_coords, elements, scale * u_test)
    assert np.allclose(K_scaled, scale * K, rtol=1e-12, atol=1e-12)
    u1 = np.linspace(0.1, 1.8, dof)
    u2 = np.linspace(-0.5, 0.7, dof)
    K_u1 = fcn(node_coords, elements, u1)
    K_u2 = fcn(node_coords, elements, u2)
    K_super = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_super, K_u1 + K_u2, rtol=1e-12, atol=1e-12)
    elements_reversed = list(reversed(elements))
    K_rev = fcn(node_coords, elements_reversed, u_test)
    assert np.allclose(K_rev, K, rtol=1e-12, atol=1e-12)

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
        t = pj - pi
        L = np.linalg.norm(t)
        if L == 0.0:
            R = np.eye(3)
        else:
            x_hat = t / L
            if local_z is None:
                v = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                v = np.array(local_z, dtype=float)
            v = v / (np.linalg.norm(v) + 1e-15)
            z_temp = v - np.dot(v, x_hat) * x_hat
            nz = np.linalg.norm(z_temp)
            if nz < 1e-12:
                v = np.array([0.0, 1.0, 0.0], dtype=float)
                z_temp = v - np.dot(v, x_hat) * x_hat
                nz = np.linalg.norm(z_temp)
                if nz < 1e-12:
                    R = np.eye(3)
                else:
                    z_hat = z_temp / nz
                    y_hat = np.cross(z_hat, x_hat)
                    y_hat = y_hat / (np.linalg.norm(y_hat) + 1e-15)
                    z_hat = np.cross(x_hat, y_hat)
                    R = np.vstack([x_hat, y_hat, z_hat])
            else:
                z_hat = z_temp / nz
                y_hat = np.cross(z_hat, x_hat)
                y_hat = y_hat / (np.linalg.norm(y_hat) + 1e-15)
                z_hat = np.cross(x_hat, y_hat)
                R = np.vstack([x_hat, y_hat, z_hat])
        R6 = np.zeros((6, 6), dtype=float)
        R6[:3, :3] = R
        R6[3:, 3:] = R
        Gamma = np.zeros((12, 12), dtype=float)
        Gamma[:6, :6] = R6
        Gamma[6:, 6:] = R6
        return Gamma

    def compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele.get('local_z', None))
        d_local = Gamma @ u_e_global
        weights = np.arange(1, 13, dtype=float)
        return weights * d_local

    def local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        s = Fx2 + Mx2 + My1 + Mz1 + My2 + Mz2
        return np.eye(12) * s
    g = fcn.__globals__
    g['beam_transformation_matrix_3D'] = beam_transformation_matrix_3D
    g['compute_local_element_loads_beam_3D'] = compute_local_element_loads_beam_3D
    g['local_geometric_stiffness_matrix_3D_beam'] = local_geometric_stiffness_matrix_3D_beam
    node_coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.3, -0.2], [1.8, 0.9, 0.4]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 2.0, 'I_rho': 0.8, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 1.5, 'I_rho': 1.1, 'local_z': [0.0, 0.0, 1.0]}]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u = np.linspace(-0.4, 0.9, dof)
    K = fcn(node_coords, elements, u)
    axis = np.array([0.3, -0.7, 0.6], dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-15)
    theta = 0.7
    K_axis = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=float)
    R = np.eye(3) + np.sin(theta) * K_axis + (1 - np.cos(theta)) * (K_axis @ K_axis)
    R6 = np.zeros((6, 6), dtype=float)
    R6[:3, :3] = R
    R6[3:, 3:] = R
    T = np.zeros((dof, dof), dtype=float)
    for i in range(n_nodes):
        T[i * 6:(i + 1) * 6, i * 6:(i + 1) * 6] = R6
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    for ele in elements:
        lz = np.array(ele['local_z'], dtype=float)
        elements_rot.append({'node_i': ele['node_i'], 'node_j': ele['node_j'], 'A': ele['A'], 'I_rho': ele['I_rho'], 'local_z': (R @ lz).tolist()})
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    assert K.shape == (dof, dof)
    assert K_rot.shape == (dof, dof)
    TK = T @ K @ T.T
    assert np.allclose(K_rot, TK, rtol=1e-11, atol=1e-11)