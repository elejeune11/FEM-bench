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
        xi_vec = np.array([xi, yi, zi], dtype=float)
        xj_vec = np.array([xj, yj, zj], dtype=float)
        ex = xj_vec - xi_vec
        L = np.linalg.norm(ex)
        if L == 0.0:
            raise ValueError('Zero-length element')
        ex = ex / L
        if local_z is None:
            z_guess = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(ex, z_guess)) > 0.999:
                z_guess = np.array([0.0, 1.0, 0.0])
        else:
            z_guess = np.array(local_z, dtype=float)
            nz = np.linalg.norm(z_guess)
            if nz == 0.0 or abs(np.dot(ex, z_guess / nz)) > 0.999:
                z_guess = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(ex, z_guess)) > 0.999:
                    z_guess = np.array([0.0, 1.0, 0.0])
        z_guess = z_guess / np.linalg.norm(z_guess)
        ey = np.cross(z_guess, ex)
        ey_norm = np.linalg.norm(ey)
        if ey_norm == 0.0:
            alt = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(ex, alt)) > 0.999:
                alt = np.array([1.0, 0.0, 0.0])
            ey = np.cross(alt, ex)
            ey = ey / np.linalg.norm(ey)
        else:
            ey = ey / ey_norm
        ez = np.cross(ex, ey)
        Q = np.column_stack([ex, ey, ez])
        blocks = [Q.T, Q.T, Q.T, Q.T]
        Gamma = np.zeros((12, 12), dtype=float)
        for k in range(4):
            Gamma[3 * k:3 * (k + 1), 3 * k:3 * (k + 1)] = blocks[k]
        return Gamma

    def compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele.get('local_z', None))
        d_local = Gamma @ u_e_global
        weights = np.array([2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3], dtype=float)
        scale = float(ele.get('A', 1.0)) + float(ele.get('I_rho', 0.0))
        return scale * (weights * d_local)

    def local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        Ls = max(float(L), 1e-12)
        alpha = A / Ls
        beta = I_rho / Ls
        gamma = 0.5 * (A + I_rho) / Ls
        P_t = np.diag([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]).astype(float)
        P_rx = np.diag([0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]).astype(float)
        P_ry = np.diag([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]).astype(float)
        P_rz = np.diag([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]).astype(float)
        k = alpha * Fx2 * P_t + beta * Mx2 * P_rx + gamma * ((My1 + My2) * P_ry + (Mz1 + Mz2) * P_rz)
        k = 0.5 * (k + k.T)
        return k
    fcn.__globals__['beam_transformation_matrix_3D'] = beam_transformation_matrix_3D
    fcn.__globals__['compute_local_element_loads_beam_3D'] = compute_local_element_loads_beam_3D
    fcn.__globals__['local_geometric_stiffness_matrix_3D_beam'] = local_geometric_stiffness_matrix_3D_beam
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 2.0, 'I_rho': 0.5, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 1.5, 'I_rho': 0.7, 'local_z': [0.0, 0.0, 1.0]}]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u_zero = np.zeros(dof, dtype=float)
    K_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_zero, np.zeros((dof, dof)))
    u1 = np.array([0.1, -0.2, 0.3, 0.01, -0.02, 0.03, -0.4, 0.5, -0.6, 0.04, -0.05, 0.06, 0.7, -0.8, 0.9, -0.07, 0.08, -0.09], dtype=float)
    K1 = fcn(node_coords, elements, u1)
    assert K1.shape == (dof, dof)
    assert np.allclose(K1, K1.T, atol=1e-12)
    alpha = 3.4
    K1_scaled = fcn(node_coords, elements, alpha * u1)
    assert np.allclose(K1_scaled, alpha * K1, atol=1e-10, rtol=1e-10)
    u2 = np.array([-0.3, 0.25, -0.15, -0.02, 0.01, -0.04, 0.2, -0.1, 0.05, -0.03, 0.02, -0.01, -0.05, 0.15, -0.25, 0.06, -0.07, 0.08], dtype=float)
    K2 = fcn(node_coords, elements, u2)
    K_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_sum, K1 + K2, atol=1e-10, rtol=1e-10)
    K1_rev = fcn(node_coords, list(reversed(elements)), u1)
    assert np.allclose(K1_rev, K1, atol=1e-12, rtol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """

    def beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        xi_vec = np.array([xi, yi, zi], dtype=float)
        xj_vec = np.array([xj, yj, zj], dtype=float)
        ex = xj_vec - xi_vec
        L = np.linalg.norm(ex)
        if L == 0.0:
            raise ValueError('Zero-length element')
        ex = ex / L
        if local_z is None:
            z_guess = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(ex, z_guess)) > 0.999:
                z_guess = np.array([0.0, 1.0, 0.0])
        else:
            z_guess = np.array(local_z, dtype=float)
            nz = np.linalg.norm(z_guess)
            if nz == 0.0 or abs(np.dot(ex, z_guess / nz)) > 0.999:
                z_guess = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(ex, z_guess)) > 0.999:
                    z_guess = np.array([0.0, 1.0, 0.0])
        z_guess = z_guess / np.linalg.norm(z_guess)
        ey = np.cross(z_guess, ex)
        ey = ey / np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        Q = np.column_stack([ex, ey, ez])
        blocks = [Q.T, Q.T, Q.T, Q.T]
        Gamma = np.zeros((12, 12), dtype=float)
        for k in range(4):
            Gamma[3 * k:3 * (k + 1), 3 * k:3 * (k + 1)] = blocks[k]
        return Gamma

    def compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele.get('local_z', None))
        d_local = Gamma @ u_e_global
        weights = np.array([2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3], dtype=float)
        scale = float(ele.get('A', 1.0)) + float(ele.get('I_rho', 0.0))
        return scale * (weights * d_local)

    def local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        Ls = max(float(L), 1e-12)
        alpha = A / Ls
        beta = I_rho / Ls
        gamma = 0.5 * (A + I_rho) / Ls
        P_t = np.diag([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]).astype(float)
        P_rx = np.diag([0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]).astype(float)
        P_ry = np.diag([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]).astype(float)
        P_rz = np.diag([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]).astype(float)
        k = alpha * Fx2 * P_t + beta * Mx2 * P_rx + gamma * ((My1 + My2) * P_ry + (Mz1 + Mz2) * P_rz)
        k = 0.5 * (k + k.T)
        return k
    fcn.__globals__['beam_transformation_matrix_3D'] = beam_transformation_matrix_3D
    fcn.__globals__['compute_local_element_loads_beam_3D'] = compute_local_element_loads_beam_3D
    fcn.__globals__['local_geometric_stiffness_matrix_3D_beam'] = local_geometric_stiffness_matrix_3D_beam
    node_coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.2, 0.8, 0.0]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.8, 'I_rho': 0.4, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 2.2, 'I_rho': 0.6, 'local_z': [0.0, 0.0, 1.0]}]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u = np.array([0.15, -0.25, 0.35, 0.02, -0.03, 0.04, -0.45, 0.55, -0.65, -0.05, 0.06, -0.07, 0.75, -0.85, 0.95, 0.08, -0.09, 0.11], dtype=float)
    K = fcn(node_coords, elements, u)
    theta = 0.3
    R = np.array([[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]], dtype=float)
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for ele in elements:
        z = np.array(ele['local_z'], dtype=float)
        z_rot = z @ R.T
        ele_rot = dict(ele)
        ele_rot['local_z'] = z_rot.tolist()
        elements_rot.append(ele_rot)
    T = np.zeros((dof, dof), dtype=float)
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_expected = T @ K @ T.T
    assert K.shape == (dof, dof)
    assert K_rot.shape == (dof, dof)
    assert np.allclose(K_rot, K_expected, atol=1e-10, rtol=1e-10)