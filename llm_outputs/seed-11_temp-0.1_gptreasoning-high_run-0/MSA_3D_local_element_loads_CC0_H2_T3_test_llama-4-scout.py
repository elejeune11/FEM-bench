def test_rigid_body_motion_zero_loads(fcn):
    """
    Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector.
    """
    ele_info = {'E': 1.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.01, 'I_z': 0.01, 'J': 0.01}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_dofs_global = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(load_dofs_local, np.zeros(12))

def test_unit_responses_axial_shear_torsion(fcn):
    """
    Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation
    """
    ele_info = {'E': 1.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.01, 'I_z': 0.01, 'J': 0.01}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_dofs_global = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(load_dofs_local[:3], np.array([ele_info['E'] * ele_info['A'], 0, 0]))
    assert np.allclose(load_dofs_local[3:6], np.zeros(3))
    assert np.allclose(load_dofs_local[6:9], np.array([-ele_info['E'] * ele_info['A'], 0, 0]))
    assert np.allclose(load_dofs_local[9:], np.zeros(3))
    u_dofs_global = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(load_dofs_local[:3], np.zeros(3))
    assert np.allclose(load_dofs_local[3:6], np.zeros(3))
    assert np.allclose(load_dofs_local[6:9], np.zeros(3))
    assert np.allclose(load_dofs_local[9:12], np.zeros(3))
    u_dofs_global = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(load_dofs_local[:3], np.zeros(3))
    assert np.allclose(load_dofs_local[3:6], np.array([0.0, 0.0, ele_info['G'] * ele_info['J']]))
    assert np.allclose(load_dofs_local[6:9], np.zeros(3))
    assert np.allclose(load_dofs_local[9:12], np.array([0.0, 0.0, -ele_info['G'] * ele_info['J']]))

def test_superposition_linearity(fcn):
    """
    Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds.
    """
    ele_info = {'E': 1.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.01, 'I_z': 0.01, 'J': 0.01}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_dofs_global_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u_dofs_global_b = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    load_dofs_local_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_a)
    load_dofs_local_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_b)
    load_dofs_local_sum = load_dofs_local_a + load_dofs_local_b
    load_dofs_local_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_a + u_dofs_global_b)
    assert np.allclose(load_dofs_local_sum, load_dofs_local_combined)

def test_coordinate_invariance_global_rotation(fcn):
    """
    Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged.
    """
    ele_info = {'E': 1.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.01, 'I_z': 0.01, 'J': 0.01, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_dofs_global = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    load_dofs_local_original = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    (xi_rotated, yi_rotated, zi_rotated) = np.dot(R, np.array([xi, yi, zi]))
    (xj_rotated, yj_rotated, zj_rotated) = np.dot(R, np.array([xj, yj, zj]))
    u_dofs_global_rotated = np.hstack([np.dot(R, u_dofs_global[:3]), np.dot(R, u_dofs_global[3:6]), np.dot(R, u_dofs_global[6:9]), np.dot(R, u_dofs_global[9:12])])
    ele_info_rotated = ele_info.copy()
    ele_info_rotated['local_z'] = np.dot(R, ele_info['local_z'])
    load_dofs_local_rotated = fcn(ele_info_rotated, xi_rotated, yi_rotated, zi_rotated, xj_rotated, yj_rotated, zj_rotated, u_dofs_global_rotated)
    assert np.allclose(load_dofs_local_original, load_dofs_local_rotated)