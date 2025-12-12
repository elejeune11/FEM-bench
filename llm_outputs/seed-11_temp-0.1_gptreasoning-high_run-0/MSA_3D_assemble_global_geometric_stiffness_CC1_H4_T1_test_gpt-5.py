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
        dx = xj - xi
        dy = yj - yi
        dz = zj - zi
        L = np.sqrt(dx * dx + dy * dy + dz * dz)
        ex = np.array([dx, dy, dz], dtype=float) / L
        v = np.array(local_z if local_z is not None else [0.0, 0.0, 1.0], dtype=float)
        if np.linalg.norm(v) < 1e-14:
            v = np.array([0.0, 0.0, 1.0], dtype=float)
        if np.linalg.norm(np.cross(ex, v)) < 1e-12:
            v = np.array([0.0, 1.0, 0.0], dtype=float)
        ez = np.cross(ex, v)
        ez = ez / np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        ey = ey / np.linalg.norm(ey)
        R = np.column_stack((ex, ey, ez))
        Gamma = np.zeros((12, 12), dtype=float)
        for k in range(4):
            Gamma[3 * k:3 * k + 3, 3 * k:3 * k + 3] = R.T
        return Gamma

    def compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele.get('local_z', None))
        d_l = Gamma @ u_e_global
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        A = float(ele['A'])
        I_rho = float(ele['I_rho'])
        k_t = A / (L + 1.0) + 1.0
        k_r = I_rho / (L + 1.0) + 2.0
        f_l = np.zeros(12, dtype=float)
        f_l[0:3] = k_t * d_l[0:3]
        f_l[6:9] = k_t * d_l[6:9]
        f_l[3:6] = k_r * d_l[3:6]
        f_l[9:12] = k_r * d_l[9:12]
        return f_l

    def local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        coeff = (A / (L + 1.0) + 0.1 * I_rho) * (Fx2 + 0.5 * Mx2 + 0.3 * (My1 + Mz1 + My2 + Mz2))
        w = np.linspace(1.0, 2.2, 12)
        M0 = np.diag(w) + 0.05 * (np.ones((12, 12)) - np.eye(12))
        return coeff * M0
    fcn.__globals__['beam_transformation_matrix_3D'] = beam_transformation_matrix_3D
    fcn.__globals__['compute_local_element_loads_beam_3D'] = compute_local_element_loads_beam_3D
    fcn.__globals__['local_geometric_stiffness_matrix_3D_beam'] = local_geometric_stiffness_matrix_3D_beam
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 2.0, 'I_rho': 3.0, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 1, 'node_j': 2, 'A': 2.5, 'I_rho': 1.5, 'local_z': np.array([0.0, 0.0, 1.0])}]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u_zero = np.zeros(dof, dtype=float)
    K_zero = fcn(node_coords, elements, u_zero)
    assert isinstance(K_zero, np.ndarray)
    assert K_zero.shape == (dof, dof)
    assert np.allclose(K_zero, np.zeros_like(K_zero))
    u = (np.arange(dof, dtype=float) - dof / 3.0) / 10.0
    K = fcn(node_coords, elements, u)
    assert np.allclose(K, K.T)
    alpha = 2.5
    K_scaled = fcn(node_coords, elements, alpha * u)
    assert np.allclose(K_scaled, alpha * K, rtol=1e-12, atol=1e-12)
    u1 = np.zeros(dof, dtype=float)
    u2 = np.zeros(dof, dtype=float)
    u1[1] = 0.2
    u1[6 + 3] = -0.2
    u1[12 + 2] = 0.3
    u2[0] = -0.4
    u2[12 + 5] = 0.5
    u2[6 + 4] = -0.1
    K_super = fcn(node_coords, elements, u1 + u2)
    K_sum = fcn(node_coords, elements, u1) + fcn(node_coords, elements, u2)
    assert np.allclose(K_super, K_sum, rtol=1e-12, atol=1e-12)
    elements_reversed = list(reversed(elements))
    K_rev = fcn(node_coords, elements_reversed, u)
    assert np.allclose(K, K_rev, rtol=1e-12, atol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """

    def beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        dx = xj - xi
        dy = yj - yi
        dz = zj - zi
        L = np.sqrt(dx * dx + dy * dy + dz * dz)
        ex = np.array([dx, dy, dz], dtype=float) / L
        v = np.array(local_z if local_z is not None else [0.0, 0.0, 1.0], dtype=float)
        if np.linalg.norm(v) < 1e-14:
            v = np.array([0.0, 0.0, 1.0], dtype=float)
        if np.linalg.norm(np.cross(ex, v)) < 1e-12:
            v = np.array([0.0, 1.0, 0.0], dtype=float)
        ez = np.cross(ex, v)
        ez = ez / np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        ey = ey / np.linalg.norm(ey)
        R = np.column_stack((ex, ey, ez))
        Gamma = np.zeros((12, 12), dtype=float)
        for k in range(4):
            Gamma[3 * k:3 * k + 3, 3 * k:3 * k + 3] = R.T
        return Gamma

    def compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele.get('local_z', None))
        d_l = Gamma @ u_e_global
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        A = float(ele['A'])
        I_rho = float(ele['I_rho'])
        k_t = A / (L + 1.0) + 1.0
        k_r = I_rho / (L + 1.0) + 2.0
        f_l = np.zeros(12, dtype=float)
        f_l[0:3] = k_t * d_l[0:3]
        f_l[6:9] = k_t * d_l[6:9]
        f_l[3:6] = k_r * d_l[3:6]
        f_l[9:12] = k_r * d_l[9:12]
        return f_l

    def local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        coeff = (A / (L + 1.0) + 0.1 * I_rho) * (Fx2 + 0.5 * Mx2 + 0.3 * (My1 + Mz1 + My2 + Mz2))
        w = np.linspace(1.0, 2.2, 12)
        M0 = np.diag(w) + 0.05 * (np.ones((12, 12)) - np.eye(12))
        return coeff * M0
    fcn.__globals__['beam_transformation_matrix_3D'] = beam_transformation_matrix_3D
    fcn.__globals__['compute_local_element_loads_beam_3D'] = compute_local_element_loads_beam_3D
    fcn.__globals__['local_geometric_stiffness_matrix_3D_beam'] = local_geometric_stiffness_matrix_3D_beam
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 2.0, 'I_rho': 3.0, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 1, 'node_j': 2, 'A': 2.5, 'I_rho': 1.5, 'local_z': np.array([0.0, 0.0, 1.0])}]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u = (np.arange(dof, dtype=float) + 1.0) / 17.0
    K = fcn(node_coords, elements, u)
    axis = np.array([0.3, -0.4, 0.5], dtype=float)
    axis = axis / np.linalg.norm(axis)
    angle = 0.7
    Kx = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]], dtype=float)
    R0 = np.eye(3) + np.sin(angle) * Kx + (1.0 - np.cos(angle)) * (Kx @ Kx)
    T = np.zeros((dof, dof), dtype=float)
    for n in range(n_nodes):
        T[6 * n:6 * n + 3, 6 * n:6 * n + 3] = R0
        T[6 * n + 3:6 * n + 6, 6 * n + 3:6 * n + 6] = R0
    node_coords_rot = (R0 @ node_coords.T).T
    elements_rot = []
    for ele in elements:
        z0 = np.array(ele.get('local_z', [0.0, 0.0, 1.0]), dtype=float)
        elements_rot.append({**ele, 'local_z': R0 @ z0})
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    assert np.allclose(K_rot, T @ K @ T.T, rtol=1e-11, atol=1e-11)