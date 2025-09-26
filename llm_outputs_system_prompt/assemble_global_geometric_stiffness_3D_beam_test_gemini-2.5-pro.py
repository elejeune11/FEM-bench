def test_multi_element_core_correctness_assembly(fcn, monkeypatch):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
for a simple 3-node, 2-element chain. Checks that:
  1) zero displacement produces a zero matrix,
  2) the assembled matrix is symmetric,
  3) scaling displacements scales K_g linearly,
  4) superposition holds for independent displacement states, and
  5) element order does not affect the assembled result."""

    def mock_beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        R = np.eye(3)
        Gamma = np.kron(np.eye(4), R)
        return Gamma

    def mock_compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        Fx2 = u_e_global[6] - u_e_global[0]
        My1 = u_e_global[4]
        Mz2 = u_e_global[11]
        f_local = np.zeros(12)
        f_local[0] = -Fx2
        f_local[4] = My1
        f_local[6] = Fx2
        f_local[11] = Mz2
        return f_local

    def mock_local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        k_g = np.zeros((12, 12))
        if L > 1e-09:
            k_g[1, 1] = Fx2 / L
            k_g[7, 7] = Fx2 / L
            k_g[1, 7] = k_g[7, 1] = -Fx2 / L
            k_g[4, 10] = k_g[10, 4] = My1 / L
            k_g[5, 11] = k_g[11, 5] = Mz2 / L
        return k_g
    module = fcn.__module__
    monkeypatch.setattr(f'{module}.beam_transformation_matrix_3D', mock_beam_transformation_matrix_3D)
    monkeypatch.setattr(f'{module}.compute_local_element_loads_beam_3D', mock_compute_local_element_loads_beam_3D)
    monkeypatch.setattr(f'{module}.local_geometric_stiffness_matrix_3D_beam', mock_local_geometric_stiffness_matrix_3D_beam)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    n_nodes = len(node_coords)
    dof = 6 * n_nodes
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0}]
    u_zero = np.zeros(dof)
    K_g_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_g_zero, np.zeros((dof, dof)))
    u_global = np.zeros(dof)
    u_global[12] = 0.1
    u_global[4] = 0.05
    u_global[17] = 0.02
    K_g = fcn(node_coords, elements, u_global)
    assert np.allclose(K_g, K_g.T)
    scale = 2.5
    u_scaled = u_global * scale
    K_g_scaled = fcn(node_coords, elements, u_scaled)
    assert np.allclose(K_g_scaled, K_g * scale)
    u1 = np.zeros(dof)
    u1[12] = 0.1
    u2 = np.zeros(dof)
    u2[4] = 0.05
    u3 = np.zeros(dof)
    u3[17] = 0.02
    K_g1 = fcn(node_coords, elements, u1)
    K_g2 = fcn(node_coords, elements, u2)
    K_g3 = fcn(node_coords, elements, u3)
    K_g_sum_parts = K_g1 + K_g2 + K_g3
    u_sum = u1 + u2 + u3
    K_g_from_sum = fcn(node_coords, elements, u_sum)
    assert np.allclose(K_g_from_sum, K_g_sum_parts)
    elements_shuffled = [elements[1], elements[0]]
    K_g_shuffled = fcn(node_coords, elements_shuffled, u_global)
    assert np.allclose(K_g, K_g_shuffled)

def test_frame_objectivity_under_global_rotation(fcn, monkeypatch):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
Rotating the entire system (geometry, local axes, and displacement field) by
a global rotation R should produce a geometric stiffness matrix K_g^rot that
satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz]."""

    def real_beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        x_vec = np.array([xj - xi, yj - yi, zj - zi])
        L = np.linalg.norm(x_vec)
        if L < 1e-12:
            raise ValueError('Element length is zero.')
        x_axis = x_vec / L
        if local_z is None:
            if np.allclose(np.abs(x_axis), [0, 1, 0]):
                z_axis_approx = np.array([1.0, 0.0, 0.0])
            else:
                z_axis_approx = np.array([0.0, 0.0, 1.0])
        else:
            z_axis_approx = np.array(local_z)
        y_axis = np.cross(z_axis_approx, x_axis)
        if np.linalg.norm(y_axis) < 1e-09:
            tmp = np.array([1.0, 0.0, 0.0])
            if np.linalg.norm(np.cross(x_axis, tmp)) < 1e-09:
                tmp = np.array([0.0, 1.0, 0.0])
            y_axis = np.cross(x_axis, tmp)
        z_axis = np.cross(x_axis, y_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis /= np.linalg.norm(z_axis)
        R_gl = np.vstack([x_axis, y_axis, z_axis])
        Gamma = np.zeros((12, 12))
        Gamma[0:3, 0:3] = R_gl
        Gamma[3:6, 3:6] = R_gl
        Gamma[6:9, 6:9] = R_gl
        Gamma[9:12, 9:12] = R_gl
        return Gamma

    def mock_objective_compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        local_z = ele.get('local_z')
        Gamma = real_beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z)
        u_local = Gamma @ u_e_global
        axial_stretch = u_local[6] - u_local[0]
        Fx2 = 1000000.0 * axial_stretch
        f_local = np.zeros(12)
        f_local[6] = Fx2
        f_local[0] = -Fx2
        return f_local

    def mock_objective_local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        k_g = np.zeros((12, 12))
        if L > 1e-09:
            val = Fx2 / L
            k_g[1, 1] = k_g[7, 7] = val
            k_g[1, 7] = k_g[7, 1] = -val
            k_g[2, 2] = k_g[8, 8] = val
            k_g[2, 8] = k_g[8, 2] = -val
        return k_g
    module = fcn.__module__
    monkeypatch.setattr(f'{module}.beam_transformation_matrix_3D', real_beam_transformation_matrix_3D)
    monkeypatch.setattr(f'{module}.compute_local_element_loads_beam_3D', mock_objective_compute_local_element_loads_beam_3D)
    monkeypatch.setattr(f'{module}.local_geometric_stiffness_matrix_3D_beam', mock_objective_local_geometric_stiffness_matrix_3D_beam)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    n_nodes = len(node_coords)
    dof = 6 * n_nodes
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0]}]
    u_global = np.random.default_rng(0).random(dof) * 0.01
    K_g = fcn(node_coords, elements, u_global)

    def rotation_matrix_from_rotvec(rotvec):
        angle = np.linalg.norm(rotvec)
        if angle < 1e-12:
            return np.eye(3)
        axis = rotvec / angle
        (c, s, t) = (np.cos(angle), np.sin(angle), 1 - np.cos(angle))
        (x, y, z) = axis
        return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y], [t * x * y + s * z, t * y * y + c, t * y * z - s * x], [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])
    rotvec = np.deg2rad(30) * np.array([1, 2, 3]) / np.sqrt(14)
    R_mat = rotation_matrix_from_rotvec(rotvec)
    node_coords_rot = (R_mat @ node_coords.T).T
    elements_rot = []
    for ele in elements:
        ele_rot = ele.copy()
        if 'local_z' in ele_rot:
            ele_rot['local_z'] = R_mat @ np.array(ele_rot['local_z'])
        elements_rot.append(ele_rot)
    u_global_rot = np.zeros_like(u_global)
    for i in range(n_nodes):
        u_global_rot[6 * i:6 * i + 3] = R_mat @ u_global[6 * i:6 * i + 3]
        u_global_rot[6 * i + 3:6 * i + 6] = R_mat @ u_global[6 * i + 3:6 * i + 6]
    K_g_rot = fcn(node_coords_rot, elements_rot, u_global_rot)

    def block_diag(*arrs):
        shapes = np.array([a.shape for a in arrs])
        out = np.zeros(np.sum(shapes, axis=0), dtype=arrs[0].dtype)
        (r, c) = (0, 0)
        for (i, (rr, cc)) in enumerate(shapes):
            out[r:r + rr, c:c + cc] = arrs[i]
            r += rr
            c += cc
        return out
    R_block = block_diag(R_mat, R_mat)
    T_blocks = [R_block] * n_nodes
    T = block_diag(*T_blocks)
    K_g_transformed = T @ K_g @ T.T
    assert np.allclose(K_g_rot, K_g_transformed, atol=1e-09)