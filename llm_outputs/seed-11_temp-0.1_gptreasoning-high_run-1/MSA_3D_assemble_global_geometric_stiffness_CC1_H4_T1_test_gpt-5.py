def test_multi_element_core_correctness_assembly(fcn):
    """
    Verify basic correctness of assemble_global_geometric_stiffness_3D_beam for a simple 3-node, 2-element chain.
    Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """

    def _triad_and_gamma(xi, yi, zi, xj, yj, zj, local_z=None):
        ri = np.array([xi, yi, zi], dtype=float)
        rj = np.array([xj, yj, zj], dtype=float)
        ex = rj - ri
        L = np.linalg.norm(ex)
        if L == 0:
            raise ValueError('Zero length element')
        ex = ex / L
        a = np.array(local_z if local_z is not None else [0.0, 0.0, 1.0], dtype=float)
        if np.linalg.norm(np.cross(ex, a)) < 1e-12:
            a = np.array([0.0, 1.0, 0.0], dtype=float)
        ez = a - ex * np.dot(a, ex)
        nz = np.linalg.norm(ez)
        if nz < 1e-12:
            a = np.array([0.0, 1.0, 0.0], dtype=float)
            ez = a - ex * np.dot(a, ex)
            nz = np.linalg.norm(ez)
        ez = ez / nz
        ey = np.cross(ez, ex)
        R = np.column_stack((ex, ey, ez))
        Gamma = np.zeros((12, 12), dtype=float)
        Rt = R.T
        for b in range(4):
            Gamma[3 * b:3 * (b + 1), 3 * b:3 * (b + 1)] = Rt
        return (L, Gamma)

    def beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        _, Gamma = _triad_and_gamma(xi, yi, zi, xj, yj, zj, local_z)
        return Gamma

    def compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        _, Gamma = _triad_and_gamma(xi, yi, zi, xj, yj, zj, ele.get('local_z', None))
        u_l = Gamma @ u_e_global
        c_f = 2.0
        c_m = 3.0
        F = np.zeros(12, dtype=float)
        F[0:3] = c_f * u_l[0:3]
        F[3:6] = c_m * u_l[3:6]
        F[6:9] = c_f * u_l[6:9]
        F[9:12] = c_m * u_l[9:12]
        return F

    def local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        v_axial = np.array([1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], dtype=float)
        v_torsion = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0], dtype=float)
        v_by = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0], dtype=float)
        v_bz = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1], dtype=float)
        S1 = np.outer(v_axial, v_axial)
        S2 = np.outer(v_torsion, v_torsion)
        S3 = np.outer(v_by, v_by)
        S4 = np.outer(v_bz, v_bz)
        K = Fx2 * S1 + Mx2 * S2 + (My1 + My2) * S3 + (Mz1 + Mz2) * S4
        scale = 1.0 / (L + 1.0)
        return scale * K
    g = fcn.__globals__
    orig_btm = g.get('beam_transformation_matrix_3D', None)
    orig_cl = g.get('compute_local_element_loads_beam_3D', None)
    orig_lg = g.get('local_geometric_stiffness_matrix_3D_beam', None)
    g['beam_transformation_matrix_3D'] = beam_transformation_matrix_3D
    g['compute_local_element_loads_beam_3D'] = compute_local_element_loads_beam_3D
    g['local_geometric_stiffness_matrix_3D_beam'] = local_geometric_stiffness_matrix_3D_beam
    try:
        node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
        elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0]}]
        n_nodes = node_coords.shape[0]
        ndof = 6 * n_nodes
        u0 = np.zeros(ndof, dtype=float)
        K0 = fcn(node_coords, elements, u0)
        assert K0.shape == (ndof, ndof)
        assert np.allclose(K0, np.zeros_like(K0))
        u = np.arange(ndof, dtype=float) * 0.1
        K = fcn(node_coords, elements, u)
        assert np.allclose(K, K.T, atol=1e-12)
        alpha = 3.5
        K_alpha = fcn(node_coords, elements, alpha * u)
        assert np.allclose(K_alpha, alpha * K, atol=1e-12)
        ua = np.linspace(0.2, 1.0, ndof)
        ub = np.linspace(-0.3, 0.7, ndof)
        Kab = fcn(node_coords, elements, ua + ub)
        Ka = fcn(node_coords, elements, ua)
        Kb = fcn(node_coords, elements, ub)
        assert np.allclose(Kab, Ka + Kb, atol=1e-12)
        elements_rev = list(reversed(elements))
        K_rev = fcn(node_coords, elements_rev, u)
        assert np.allclose(K_rev, K, atol=1e-12)
    finally:
        if orig_btm is None:
            del g['beam_transformation_matrix_3D']
        else:
            g['beam_transformation_matrix_3D'] = orig_btm
        if orig_cl is None:
            del g['compute_local_element_loads_beam_3D']
        else:
            g['compute_local_element_loads_beam_3D'] = orig_cl
        if orig_lg is None:
            del g['local_geometric_stiffness_matrix_3D_beam']
        else:
            g['local_geometric_stiffness_matrix_3D_beam'] = orig_lg

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam:
    Rotating the entire system (geometry, local axes, and displacement field) by a global rotation R
    should produce K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks diag(R, R).
    """

    def _triad_and_gamma(xi, yi, zi, xj, yj, zj, local_z=None):
        ri = np.array([xi, yi, zi], dtype=float)
        rj = np.array([xj, yj, zj], dtype=float)
        ex = rj - ri
        L = np.linalg.norm(ex)
        if L == 0:
            raise ValueError('Zero length element')
        ex = ex / L
        a = np.array(local_z if local_z is not None else [0.0, 0.0, 1.0], dtype=float)
        if np.linalg.norm(np.cross(ex, a)) < 1e-12:
            a = np.array([0.0, 1.0, 0.0], dtype=float)
        ez = a - ex * np.dot(a, ex)
        nz = np.linalg.norm(ez)
        if nz < 1e-12:
            a = np.array([0.0, 1.0, 0.0], dtype=float)
            ez = a - ex * np.dot(a, ex)
            nz = np.linalg.norm(ez)
        ez = ez / nz
        ey = np.cross(ez, ex)
        R = np.column_stack((ex, ey, ez))
        Gamma = np.zeros((12, 12), dtype=float)
        Rt = R.T
        for b in range(4):
            Gamma[3 * b:3 * (b + 1), 3 * b:3 * (b + 1)] = Rt
        return (L, Gamma)

    def beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        _, Gamma = _triad_and_gamma(xi, yi, zi, xj, yj, zj, local_z)
        return Gamma

    def compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        _, Gamma = _triad_and_gamma(xi, yi, zi, xj, yj, zj, ele.get('local_z', None))
        u_l = Gamma @ u_e_global
        c_f = 2.0
        c_m = 3.0
        F = np.zeros(12, dtype=float)
        F[0:3] = c_f * u_l[0:3]
        F[3:6] = c_m * u_l[3:6]
        F[6:9] = c_f * u_l[6:9]
        F[9:12] = c_m * u_l[9:12]
        return F

    def local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        v_axial = np.array([1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0], dtype=float)
        v_torsion = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0], dtype=float)
        v_by = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0], dtype=float)
        v_bz = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1], dtype=float)
        S1 = np.outer(v_axial, v_axial)
        S2 = np.outer(v_torsion, v_torsion)
        S3 = np.outer(v_by, v_by)
        S4 = np.outer(v_bz, v_bz)
        K = Fx2 * S1 + Mx2 * S2 + (My1 + My2) * S3 + (Mz1 + Mz2) * S4
        scale = 1.0 / (L + 1.0)
        return scale * K
    g = fcn.__globals__
    orig_btm = g.get('beam_transformation_matrix_3D', None)
    orig_cl = g.get('compute_local_element_loads_beam_3D', None)
    orig_lg = g.get('local_geometric_stiffness_matrix_3D_beam', None)
    g['beam_transformation_matrix_3D'] = beam_transformation_matrix_3D
    g['compute_local_element_loads_beam_3D'] = compute_local_element_loads_beam_3D
    g['local_geometric_stiffness_matrix_3D_beam'] = local_geometric_stiffness_matrix_3D_beam
    try:
        node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.3], [2.0, -0.1, 0.4]], dtype=float)
        elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0]}]
        n_nodes = node_coords.shape[0]
        ndof = 6 * n_nodes
        u = np.linspace(-0.4, 0.6, ndof)
        ax = 0.3
        ay = -0.25
        az = 0.4
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(ax), -np.sin(ax)], [0.0, np.sin(ax), np.cos(ax)]], dtype=float)
        Ry = np.array([[np.cos(ay), 0.0, np.sin(ay)], [0.0, 1.0, 0.0], [-np.sin(ay), 0.0, np.cos(ay)]], dtype=float)
        Rz = np.array([[np.cos(az), -np.sin(az), 0.0], [np.sin(az), np.cos(az), 0.0], [0.0, 0.0, 1.0]], dtype=float)
        R = Rz @ Ry @ Rx
        K = fcn(node_coords, elements, u)
        node_coords_rot = (R @ node_coords.T).T
        z0 = np.array([0.0, 0.0, 1.0])
        z_rot = (R @ z0).tolist()
        elements_rot = [{**ele, 'local_z': z_rot} for ele in elements]
        Tn = np.zeros((6, 6), dtype=float)
        Tn[:3, :3] = R
        Tn[3:6, 3:6] = R
        T_big = np.kron(np.eye(n_nodes), Tn)
        u_rot = T_big @ u
        K_rot = fcn(node_coords_rot, elements_rot, u_rot)
        assert np.allclose(K_rot, T_big @ K @ T_big.T, atol=1e-12)
    finally:
        if orig_btm is None:
            del g['beam_transformation_matrix_3D']
        else:
            g['beam_transformation_matrix_3D'] = orig_btm
        if orig_cl is None:
            del g['compute_local_element_loads_beam_3D']
        else:
            g['compute_local_element_loads_beam_3D'] = orig_cl
        if orig_lg is None:
            del g['local_geometric_stiffness_matrix_3D_beam']
        else:
            g['local_geometric_stiffness_matrix_3D_beam'] = orig_lg