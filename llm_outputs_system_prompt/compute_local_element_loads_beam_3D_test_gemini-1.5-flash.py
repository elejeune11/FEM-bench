def test_rigid_body_motion_zero_loads(fcn):
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    u_dofs_global = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert_allclose(loads, 0, atol=1e-10)

def test_unit_responses_axial_shear_torsion(fcn):
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    u_dofs_global = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    loads_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert_allclose(loads_axial[0], ele_info['E'] * ele_info['A'], rtol=1e-05)
    assert_allclose(loads_axial[6], -ele_info['E'] * ele_info['A'], rtol=1e-05)
    u_dofs_global = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    loads_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert_allclose(loads_shear[1], 12 * ele_info['E'] * ele_info['I_y'], rtol=1e-05)
    assert_allclose(loads_shear[7], -12 * ele_info['E'] * ele_info['I_y'], rtol=1e-05)
    u_dofs_global = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    loads_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert_allclose(loads_torsion[5], 4 * ele_info['E'] * ele_info['I_y'], rtol=1e-05)
    assert_allclose(loads_torsion[11], 4 * ele_info['E'] * ele_info['I_y'], rtol=1e-05)

def test_superposition_linearity(fcn):
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    u_a = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    u_b = np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    loads_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_a)
    loads_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_b)
    loads_ab = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_a + u_b)
    assert_allclose(loads_ab, loads_a + loads_b, rtol=1e-05)

def test_coordinate_invariance_global_rotation(fcn):
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    u_dofs_global = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    (xi_r, yi_r, zi_r) = R @ np.array([xi, yi, zi])
    (xj_r, yj_r, zj_r) = R @ np.array([xj, yj, zj])
    local_z_r = R @ ele_info['local_z']
    ele_info_r = {'E': ele_info['E'], 'nu': ele_info['nu'], 'A': ele_info['A'], 'I_y': ele_info['I_y'], 'I_z': ele_info['I_z'], 'J': ele_info['J'], 'local_z': local_z_r}
    u_dofs_global_r = np.zeros(12)
    u_dofs_global_r[:3] = R @ u_dofs_global[:3]
    u_dofs_global_r[3:6] = R @ u_dofs_global[3:6]
    u_dofs_global_r[6:9] = R @ u_dofs_global[6:9]
    u_dofs_global_r[9:] = R @ u_dofs_global[9:]
    loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    loads_r = fcn(ele_info_r, xi_r, yi_r, zi_r, xj_r, yj_r, zj_r, u_dofs_global_r)
    assert_allclose(loads, loads_r, rtol=1e-05)