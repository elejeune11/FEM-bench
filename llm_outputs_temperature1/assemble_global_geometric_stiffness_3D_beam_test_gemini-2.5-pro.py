def test_multi_element_core_correctness_assembly(fcn, monkeypatch):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result."""
    n_nodes = 3
    dofs_per_node = 6
    total_dofs = n_nodes * dofs_per_node
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0}, {'node_i': 1, 'node_j': 2, 'A': 2.0, 'I_rho': 2.0}]
    mock_gamma = np.eye(12)
    monkeypatch.setattr(f'{fcn.__module__}.beam_transformation_matrix_3D', lambda *args, **kwargs: mock_gamma)

    def mock_compute_loads(ele: Dict[str, Any], xi: float, yi: float, zi: float, xj: float, yj: float, zj: float, u_e_global: np.ndarray) -> np.ndarray:
        L = np.linalg.norm(np.array([xj, yj, zj]) - np.array([xi, yi, zi]))
        axial_stretch = u_e_global[6] - u_e_global[0]
        axial_force = ele['A'] / L * axial_stretch
        loads = np.zeros(12)
        loads[0] = -axial_force
        loads[6] = axial_force
        return loads
    monkeypatch.setattr(f'{fcn.__module__}.compute_local_element_loads_beam_3D', mock_compute_loads)
    M = np.arange(144, dtype=float).reshape(12, 12)
    M = M + M.T

    def mock_local_kg(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        return Fx2 / L * M
    monkeypatch.setattr(f'{fcn.__module__}.local_geometric_stiffness_matrix_3D_beam', mock_local_kg)
    u_zero = np.zeros(total_dofs)
    K_g_zero = fcn(node_coords, elements, u_zero)
    assert np.all(K_g_zero == 0)
    np.random.seed(0)
    u_1 = np.random.rand(total_dofs)
    K_g_1 = fcn(node_coords, elements, u_1)
    assert np.allclose(K_g_1, K_g_1.T)
    scale = 2.5
    u_scaled = u_1 * scale
    K_g_scaled = fcn(node_coords, elements, u_scaled)
    assert np.allclose(K_g_scaled, K_g_1 * scale)
    u_2 = np.random.rand(total_dofs)
    K_g_2 = fcn(node_coords, elements, u_2)
    u_sum = u_1 + u_2
    K_g_sum = fcn(node_coords, elements, u_sum)
    assert np.allclose(K_g_sum, K_g_1 + K_g_2)
    elements_rev = list(reversed(elements))
    K_g_rev = fcn(node_coords, elements_rev, u_1)
    assert np.allclose(K_g_1, K_g_rev)

def test_frame_objectivity_under_global_rotation(fcn, monkeypatch):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz]."""
    n_nodes = 2
    total_dofs = n_nodes * 6
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.5, 'I_rho': 0.5, 'local_z': [0.0, 0.0, 1.0]}]

    def mock_beam_transformation(xi, yi, zi, xj, yj, zj, local_z=None):
        p1 = np.array([xi, yi, zi])
        p2 = np.array([xj, yj, zj])
        vec_x = p2 - p1
        vec_x /= np.linalg.norm(vec_x)
        vec_z_approx = np.array(local_z)
        vec_y = np.cross(vec_z_approx, vec_x)
        vec_y /= np.linalg.norm(vec_y)
        vec_z = np.cross(vec_x, vec_y)
        vec_z /= np.linalg.norm(vec_z)
        R_local_to_global = np.vstack([vec_x, vec_y, vec_z]).T
        R_global_to_local = R_local_to_global.T
        Gamma = np.zeros((12, 12))
        for i in range(4):
            Gamma[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = R_global_to_local
        return Gamma
    monkeypatch.setattr(f'{fcn.__module__}.beam_transformation_matrix_3D', mock_beam_transformation)

    def mock_compute_loads(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        Gamma = mock_beam_transformation(xi, yi, zi, xj, yj, zj, ele.get('local_z'))
        u_e_local = Gamma @ u_e_global
        L = np.linalg.norm(np.array([xj, yj, zj]) - np.array([xi, yi, zi]))
        axial_stretch = u_e_local[6] - u_e_local[0]
        axial_force = ele['A'] / L * axial_stretch
        loads = np.zeros(12)
        loads[6] = axial_force
        loads[0] = -axial_force
        return loads
    monkeypatch.setattr(f'{fcn.__module__}.compute_local_element_loads_beam_3D', mock_compute_loads)
    M = np.arange(144, dtype=float).reshape(12, 12)
    M = M + M.T

    def mock_local_kg(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        return Fx2 / L * M if L > 1e-09 else np.zeros((12, 12))
    monkeypatch.setattr(f'{fcn.__module__}.local_geometric_stiffness_matrix_3D_beam', mock_local_kg)
    theta = np.pi / 2
    (c, s) = (np.cos(theta), np.sin(theta))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    np.random.seed(1)
    u_global = np.random.rand(total_dofs)
    K_g = fcn(node_coords, elements, u_global)
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = [elements[0].copy()]
    elements_rot[0]['local_z'] = R @ elements[0]['local_z']
    u_global_rot = np.zeros_like(u_global)
    for i in range(n_nodes):
        u_i = u_global[i * 6:i * 6 + 3]
        th_i = u_global[i * 6 + 3:i * 6 + 6]
        u_global_rot[i * 6:i * 6 + 3] = R @ u_i
        u_global_rot[i * 6 + 3:i * 6 + 6] = R @ th_i
    K_g_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
    T_node = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
    T = np.kron(np.eye(n_nodes, dtype=float), T_node)
    K_g_transformed = T @ K_g @ T.T
    assert np.allclose(K_g_rot, K_g_transformed)