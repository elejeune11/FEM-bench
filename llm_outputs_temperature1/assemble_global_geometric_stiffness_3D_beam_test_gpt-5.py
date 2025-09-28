def test_multi_element_core_correctness_assembly(fcn):
    """
    Verify basic correctness of assemble_global_geometric_stiffness_3D_beam for a simple
    3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """

    def _rot_from_coords(xi, yi, zi, xj, yj, zj, local_z=None):
        dx = np.array([xj - xi, yj - yi, zj - zi], dtype=float)
        L = np.linalg.norm(dx)
        if L == 0:
            raise ValueError('Zero-length element')
        ex = dx / L
        if local_z is not None:
            z_approx = np.asarray(local_z, dtype=float)
            nz = np.linalg.norm(z_approx)
            if nz < 1e-12:
                z_approx = np.array([0.0, 0.0, 1.0])
            else:
                z_approx = z_approx / nz
        else:
            z_approx = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z_approx, ex)) > 0.99:
            z_approx = np.array([0.0, 1.0, 0.0])
        ey = np.cross(z_approx, ex)
        ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        R = np.column_stack((ex, ey, ez))
        return R

    def beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        R = _rot_from_coords(xi, yi, zi, xj, yj, zj, local_z)
        Gamma = np.zeros((12, 12), dtype=float)
        Rt = R.T
        Gamma[0:3, 0:3] = Rt
        Gamma[3:6, 3:6] = Rt
        Gamma[6:9, 6:9] = Rt
        Gamma[9:12, 9:12] = Rt
        return Gamma

    def compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele.get('local_z', None))
        d_l = Gamma @ u_e_global
        weights = np.arange(1.0, 13.0)
        C = 1.0 + np.linalg.norm([xj - xi, yj - yi, zj - zi])
        q_l = C * weights * d_l
        return q_l

    def local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        s = Fx2 + Mx2 + My1 + Mz1 + My2 + Mz2
        S0 = np.diag(np.arange(1.0, 13.0))
        return s * S0
    g = fcn.__globals__
    g['beam_transformation_matrix_3D'] = beam_transformation_matrix_3D
    g['compute_local_element_loads_beam_3D'] = compute_local_element_loads_beam_3D
    g['local_geometric_stiffness_matrix_3D_beam'] = local_geometric_stiffness_matrix_3D_beam
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0]}]
    elements_reversed = elements[::-1]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u_zero = np.zeros(dof)
    K_zero = fcn(node_coords, elements, u_zero)
    assert K_zero.shape == (dof, dof)
    assert np.allclose(K_zero, 0.0, atol=1e-12)
    rng = np.random.default_rng(42)
    u1 = np.zeros(dof)
    u2 = np.zeros(dof)
    u1[0:6] = rng.normal(size=6)
    u2[12:18] = rng.normal(size=6)
    u = u1 + u2
    K = fcn(node_coords, elements, u)
    assert np.allclose(K, K.T, atol=1e-12)
    alpha = 3.7
    K_alpha = fcn(node_coords, elements, alpha * u)
    assert np.allclose(K_alpha, alpha * K, rtol=1e-12, atol=1e-12)
    K_u1 = fcn(node_coords, elements, u1)
    K_u2 = fcn(node_coords, elements, u2)
    assert np.allclose(K, K_u1 + K_u2, rtol=1e-12, atol=1e-12)
    K_rev = fcn(node_coords, elements_reversed, u)
    assert np.allclose(K_rev, K, rtol=1e-12, atol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce K_g^rot ≈ T K_g T^T, where T is block-diagonal with
    per-node blocks diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """

    def _rot_from_coords(xi, yi, zi, xj, yj, zj, local_z=None):
        dx = np.array([xj - xi, yj - yi, zj - zi], dtype=float)
        L = np.linalg.norm(dx)
        if L == 0:
            raise ValueError('Zero-length element')
        ex = dx / L
        if local_z is not None:
            z_approx = np.asarray(local_z, dtype=float)
            nz = np.linalg.norm(z_approx)
            if nz < 1e-12:
                z_approx = np.array([0.0, 0.0, 1.0])
            else:
                z_approx = z_approx / nz
        else:
            z_approx = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z_approx, ex)) > 0.99:
            z_approx = np.array([0.0, 1.0, 0.0])
        ey = np.cross(z_approx, ex)
        ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        R = np.column_stack((ex, ey, ez))
        return R

    def beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        R = _rot_from_coords(xi, yi, zi, xj, yj, zj, local_z)
        Gamma = np.zeros((12, 12), dtype=float)
        Rt = R.T
        Gamma[0:3, 0:3] = Rt
        Gamma[3:6, 3:6] = Rt
        Gamma[6:9, 6:9] = Rt
        Gamma[9:12, 9:12] = Rt
        return Gamma

    def compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele.get('local_z', None))
        d_l = Gamma @ u_e_global
        weights = np.arange(1.0, 13.0)
        C = 1.0 + np.linalg.norm([xj - xi, yj - yi, zj - zi])
        q_l = C * weights * d_l
        return q_l

    def local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        s = Fx2 + Mx2 + My1 + Mz1 + My2 + Mz2
        S0 = np.diag(np.arange(1.0, 13.0))
        return s * S0
    g = fcn.__globals__
    g['beam_transformation_matrix_3D'] = beam_transformation_matrix_3D
    g['compute_local_element_loads_beam_3D'] = compute_local_element_loads_beam_3D
    g['local_geometric_stiffness_matrix_3D_beam'] = local_geometric_stiffness_matrix_3D_beam
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.2], [2.0, 1.1, -0.3]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 2.3, 'I_rho': 0.9, 'local_z': [0.1, 0.3, 0.95]}, {'node_i': 1, 'node_j': 2, 'A': 1.7, 'I_rho': 1.2, 'local_z': [-0.2, 0.4, 0.88]}]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    rng = np.random.default_rng(7)
    u = rng.normal(size=dof)
    K = fcn(node_coords, elements, u)
    axis = np.array([0.3, -0.5, 0.8], dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = 0.8
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=float)
    R = np.eye(3) + np.sin(theta) * Kx + (1 - np.cos(theta)) * (Kx @ Kx)
    T = np.zeros((dof, dof), dtype=float)
    for a in range(n_nodes):
        i0 = 6 * a
        T[i0:i0 + 3, i0:i0 + 3] = R
        T[i0 + 3:i0 + 6, i0 + 3:i0 + 6] = R
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for ele in elements:
        lz = np.asarray(ele['local_z'], dtype=float)
        lz_rot = lz @ R.T
        ele_rot = dict(ele)
        ele_rot['local_z'] = lz_rot
        elements_rot.append(ele_rot)
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_pred = T @ K @ T.T
    assert np.allclose(K_rot, K_pred, rtol=1e-10, atol=1e-10)