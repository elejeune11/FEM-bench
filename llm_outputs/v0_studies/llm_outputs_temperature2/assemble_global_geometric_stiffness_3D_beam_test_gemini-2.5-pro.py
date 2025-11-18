def test_multi_element_core_correctness_assembly(fcn, monkeypatch):
    """
    Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    n_nodes = 3
    n_dof = 6 * n_nodes
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements_v1 = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.1}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 0.1}]
    elements_v2 = [elements_v1[1], elements_v1[0]]
    mock_gamma = np.eye(12)
    monkeypatch.setattr('__main__.beam_transformation_matrix_3D', lambda *args, **kwargs: mock_gamma)
    mock_kg_base_F = np.arange(144, dtype=float).reshape(12, 12)
    mock_kg_base_F = mock_kg_base_F + mock_kg_base_F.T
    mock_kg_base_M = np.eye(12) * 5.0

    def mock_local_kg(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        return Fx2 * mock_kg_base_F + My1 * mock_kg_base_M
    monkeypatch.setattr('__main__.local_geometric_stiffness_matrix_3D_beam', mock_local_kg)

    def mock_compute_loads(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        forces = np.zeros(12)
        forces[6] = 10.0 * u_e_global[6]
        forces[4] = 20.0 * u_e_global[10]
        return forces
    monkeypatch.setattr('__main__.compute_local_element_loads_beam_3D', mock_compute_loads)
    u_zero = np.zeros(n_dof)
    K_g_zero = fcn(node_coords, elements_v1, u_zero)
    assert np.allclose(K_g_zero, np.zeros((n_dof, n_dof)))
    u_a = np.zeros(n_dof)
    u_a[6] = 0.1
    u_a[12] = 0.2
    u_b = np.zeros(n_dof)
    u_b[10] = 0.3
    u_b[16] = 0.4
    K_g_a = fcn(node_coords, elements_v1, u_a)
    assert K_g_a.shape == (n_dof, n_dof)
    assert np.allclose(K_g_a, K_g_a.T)
    scale = 3.0
    u_a_scaled = u_a * scale
    K_g_a_scaled_u = fcn(node_coords, elements_v1, u_a_scaled)
    K_g_a_scaled_K = K_g_a * scale
    assert np.allclose(K_g_a_scaled_u, K_g_a_scaled_K)
    K_g_b = fcn(node_coords, elements_v1, u_b)
    u_c = u_a + u_b
    K_g_c = fcn(node_coords, elements_v1, u_c)
    assert np.allclose(K_g_c, K_g_a + K_g_b)
    K_g_v2 = fcn(node_coords, elements_v2, u_a)
    assert np.allclose(K_g_a, K_g_v2)

def test_frame_objectivity_under_global_rotation(fcn, monkeypatch):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """

    def get_rotation_matrix(axis, angle_rad):
        """Creates a 3x3 rotation matrix for rotation around a given axis."""
        axis = np.asarray(axis)
        axis = axis / np.linalg.norm(axis)
        a = np.cos(angle_rad / 2.0)
        (b, c, d) = -axis * np.sin(angle_rad / 2.0)
        (aa, bb, cc, dd) = (a * a, b * b, c * c, d * d)
        (bc, ad, ac, ab, bd, cd) = (b * c, a * d, a * c, a * b, b * d, c * d)
        return np.array([[aa + bb - cc - dd, 2 * (bc - ad), 2 * (bd + ac)], [2 * (bc + ad), aa + cc - bb - dd, 2 * (cd - ab)], [2 * (bd - ac), 2 * (cd + ab), aa + dd - bb - cc]])

    def real_beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        """A correct implementation of the 3D beam transformation matrix."""
        vec_i = np.array([xi, yi, zi])
        vec_j = np.array([xj, yj, zj])
        L = np.linalg.norm(vec_j - vec_i)
        if L < 1e-09:
            raise ValueError('Element length is zero.')
        x_prime = (vec_j - vec_i) / L
        if local_z is None:
            if np.allclose(np.abs(x_prime), [0, 1, 0]):
                ref_vec = np.array([1.0, 0.0, 0.0])
            else:
                ref_vec = np.array([0.0, 1.0, 0.0])
            z_prime = np.cross(x_prime, ref_vec)
            z_prime /= np.linalg.norm(z_prime)
            y_prime = np.cross(z_prime, x_prime)
        else:
            y_prime = np.cross(np.asarray(local_z), x_prime)
            y_prime /= np.linalg.norm(y_prime)
            z_prime = np.cross(x_prime, y_prime)
        R_3x3 = np.array([x_prime, y_prime, z_prime])
        return block_diag(R_3x3, R_3x3, R_3x3, R_3x3)
    n_nodes = 2
    n_dof = 6 * n_nodes
    node_coords = np.array([[0.5, 0.5, 0.5], [1.5, 2.5, 3.5]])
    local_z_vec = np.array([0.0, 1.0, -0.5])
    local_z_vec /= np.linalg.norm(local_z_vec)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.1, 'local_z': local_z_vec}]
    np.random.seed(0)
    u_global = np.random.rand(n_dof)
    monkeypatch.setattr('__main__.beam_transformation_matrix_3D', real_beam_transformation_matrix_3D)
    mock_local_forces = np.random.rand(12)
    monkeypatch.setattr('__main__.compute_local_element_loads_beam_3D', lambda *args, **kwargs: mock_local_forces)
    mock_kg_local_base = np.random.rand(12, 12)
    mock_kg_local = mock_kg_local_base + mock_kg_local_base.T
    monkeypatch.setattr('__main__.local_geometric_stiffness_matrix_3D_beam', lambda *args, **kwargs: mock_kg_local)
    K_g_orig = fcn(node_coords, elements, u_global)
    R = get_rotation_matrix(axis=[1, -1, 2], angle_rad=np.pi / 3)
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.1, 'local_z': R @ elements[0]['local_z']}]
    R_node = block_diag(R, R)
    T_global = block_diag(*[R_node] * n_nodes)
    u_global_rot = T_global @ u_global
    K_g_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
    K_g_expected = T_global @ K_g_orig @ T_global.T
    assert np.allclose(K_g_rot, K_g_expected, atol=1e-12)