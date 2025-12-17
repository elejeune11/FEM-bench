def test_multi_element_core_correctness_assembly(fcn):
    """
    Verify basic correctness for a simple 3-node, 2-element chain:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """

    def stub_local_geometric(*args, **kwargs):
        if len(args) >= 9:
            fx2 = float(args[3])
            mx2 = float(args[4])
            my1 = float(args[5])
            mz1 = float(args[6])
            my2 = float(args[7])
            mz2 = float(args[8])
        else:
            vals = {'Fx2': 0.0, 'Mx2': 0.0, 'My1': 0.0, 'Mz1': 0.0, 'My2': 0.0, 'Mz2': 0.0}
            for k in vals.keys():
                if k in kwargs:
                    vals[k] = float(kwargs[k])
            fx2, mx2, my1, mz1, my2, mz2 = (vals['Fx2'], vals['Mx2'], vals['My1'], vals['Mz1'], vals['My2'], vals['Mz2'])
        w = np.array([1.0, 1.4142135623730951, 3.141592653589793, 2.718281828459045, 1.618033988749895, 1.7320508075688772])
        f_vec = np.array([fx2, mx2, my1, mz1, my2, mz2])
        s = float(w @ f_vec)
        S = np.diag(np.linspace(1.0, 12.0, 12))
        return s * S
    old_dep = fcn.__globals__.get('local_geometric_stiffness_matrix_3D_beam', None)
    fcn.__globals__['local_geometric_stiffness_matrix_3D_beam'] = stub_local_geometric
    try:
        node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
        E = 210000000000.0
        nu = 0.3
        A = 0.01
        I_y = 2e-06
        I_z = 1.5e-06
        J = 1e-06
        elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}]
        n_nodes = node_coords.shape[0]
        dof = 6 * n_nodes
        u_zero = np.zeros(dof)
        K0 = fcn(node_coords, elements, u_zero)
        assert isinstance(K0, np.ndarray)
        assert K0.shape == (dof, dof)
        assert np.allclose(K0, np.zeros((dof, dof)), atol=1e-12)
        u1 = np.array([0.1, -0.05, 0.2, 0.01, -0.02, 0.03, -0.08, 0.07, -0.04, 0.02, 0.01, -0.03, 0.05, -0.06, 0.09, -0.01, 0.015, 0.02], dtype=float)
        K1 = fcn(node_coords, elements, u1)
        assert K1.shape == (dof, dof)
        assert np.allclose(K1, K1.T, atol=1e-09)
        alpha = 2.5
        K_alpha = fcn(node_coords, elements, alpha * u1)
        assert np.allclose(K_alpha, alpha * K1, rtol=1e-09, atol=1e-09)
        u2 = np.array([-0.03, 0.02, -0.01, 0.015, -0.025, 0.04, 0.06, -0.07, 0.08, -0.02, 0.03, -0.01, -0.04, 0.05, -0.06, 0.02, -0.03, 0.01], dtype=float)
        K2 = fcn(node_coords, elements, u2)
        K12 = fcn(node_coords, elements, u1 + u2)
        assert np.allclose(K12, K1 + K2, rtol=1e-09, atol=1e-09)
        elements_reversed = elements[::-1]
        K1_rev = fcn(node_coords, elements_reversed, u1)
        assert np.allclose(K1_rev, K1, rtol=1e-12, atol=1e-12)
    finally:
        if old_dep is not None:
            fcn.__globals__['local_geometric_stiffness_matrix_3D_beam'] = old_dep
        else:
            del fcn.__globals__['local_geometric_stiffness_matrix_3D_beam']

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity: rotating the entire system (geometry, local axes, displacements)
    by a global rotation R yields K_g^rot approximately equal to T K_g T^T with T block-diagonal
    diag(R,R) per node.
    """

    def stub_local_geometric(*args, **kwargs):
        if len(args) >= 9:
            fx2 = float(args[3])
            mx2 = float(args[4])
            my1 = float(args[5])
            mz1 = float(args[6])
            my2 = float(args[7])
            mz2 = float(args[8])
        else:
            vals = {'Fx2': 0.0, 'Mx2': 0.0, 'My1': 0.0, 'Mz1': 0.0, 'My2': 0.0, 'Mz2': 0.0}
            for k in vals.keys():
                if k in kwargs:
                    vals[k] = float(kwargs[k])
            fx2, mx2, my1, mz1, my2, mz2 = (vals['Fx2'], vals['Mx2'], vals['My1'], vals['Mz1'], vals['My2'], vals['Mz2'])
        w = np.array([1.0, 1.4142135623730951, 3.141592653589793, 2.718281828459045, 1.618033988749895, 1.7320508075688772])
        f_vec = np.array([fx2, mx2, my1, mz1, my2, mz2])
        s = float(w @ f_vec)
        S = np.diag(np.linspace(1.0, 12.0, 12))
        return s * S
    old_dep = fcn.__globals__.get('local_geometric_stiffness_matrix_3D_beam', None)
    fcn.__globals__['local_geometric_stiffness_matrix_3D_beam'] = stub_local_geometric
    try:

        def Rz(g):
            c, s = (np.cos(g), np.sin(g))
            return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        def Ry(b):
            c, s = (np.cos(b), np.sin(b))
            return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

        def Rx(a):
            c, s = (np.cos(a), np.sin(a))
            return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        a, b, g = (0.73, -0.51, 0.41)
        R = Rz(g) @ Ry(b) @ Rx(a)
        node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.1], [2.0, 1.0, -0.2]], dtype=float)
        node_coords_rot = node_coords @ R.T
        E = 200000000000.0
        nu = 0.29
        A = 0.012
        I_y = 1.8e-06
        I_z = 2.1e-06
        J = 1.2e-06
        local_z_vec = np.array([0.0, 0.0, 1.0])
        local_z_rot = R @ local_z_vec
        elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_vec}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_vec}]
        elements_rot = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot}]
        n_nodes = node_coords.shape[0]
        dof = 6 * n_nodes
        u = np.array([0.05, -0.02, 0.03, 0.01, -0.015, 0.02, -0.04, 0.06, -0.05, 0.02, -0.01, 0.03, 0.07, -0.03, 0.04, -0.02, 0.025, -0.015], dtype=float)
        T = np.zeros((dof, dof), dtype=float)
        for i in range(n_nodes):
            i6 = 6 * i
            T[i6:i6 + 3, i6:i6 + 3] = R
            T[i6 + 3:i6 + 6, i6 + 3:i6 + 6] = R
        u_rot = T @ u
        K = fcn(node_coords, elements, u)
        K_rot = fcn(node_coords_rot, elements_rot, u_rot)
        TKTT = T @ K @ T.T
        assert K.shape == (dof, dof)
        assert K_rot.shape == (dof, dof)
        assert np.allclose(K_rot, TKTT, rtol=5e-08, atol=5e-08)
    finally:
        if old_dep is not None:
            fcn.__globals__['local_geometric_stiffness_matrix_3D_beam'] = old_dep
        else:
            del fcn.__globals__['local_geometric_stiffness_matrix_3D_beam']