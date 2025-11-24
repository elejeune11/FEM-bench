def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    u_dofs_global = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(load_dofs_local, np.zeros(12)), 'Rigid body motion should result in zero internal loads.'

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    u_dofs_global_axial = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    load_dofs_local_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_axial)
    assert load_dofs_local_axial[0] > 0, 'Axial extension should produce axial force.'
    u_dofs_global_shear = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    load_dofs_local_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_shear)
    assert load_dofs_local_shear[1] > 0, 'Transverse shear should produce shear force.'
    u_dofs_global_torsion = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    load_dofs_local_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_torsion)
    assert load_dofs_local_torsion[3] > 0, 'Torsional rotation should produce torsional moment.'

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    u_dofs_global_a = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    u_dofs_global_b = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    load_dofs_local_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_a)
    load_dofs_local_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_b)
    u_dofs_global_combined = u_dofs_global_a + u_dofs_global_b
    load_dofs_local_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_combined)
    assert np.allclose(load_dofs_local_combined, load_dofs_local_a + load_dofs_local_b), 'Superposition does not hold.'

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    u_dofs_global = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    load_dofs_local_original = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    (xi_rot, yi_rot, zi_rot) = R @ np.array([xi, yi, zi])
    (xj_rot, yj_rot, zj_rot) = R @ np.array([xj, yj, zj])
    u_dofs_global_rot = np.zeros(12)
    u_dofs_global_rot[1] = 1
    ele_info_rot = ele_info.copy()
    ele_info_rot['local_z'] = R @ np.array(ele_info['local_z'])
    load_dofs_local_rotated = fcn(ele_info_rot, xi_rot, yi_rot, zi_rot, xj_rot, yj_rot, zj_rot, u_dofs_global_rot)
    assert np.allclose(load_dofs_local_original, load_dofs_local_rotated), 'Coordinate invariance does not hold.'