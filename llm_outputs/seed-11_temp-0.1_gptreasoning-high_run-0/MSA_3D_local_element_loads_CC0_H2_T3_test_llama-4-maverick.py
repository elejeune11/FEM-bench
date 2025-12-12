def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    u_dofs_global = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(load_dofs_local, np.zeros(12))

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses: (1) Axial unit extension, (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations, (3) Unit torsional rotation"""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    (E, A, G, J) = (ele_info['E'], ele_info['A'], ele_info['E'] / (2 * (1 + ele_info['nu'])), ele_info['J'])
    u_dofs_global_axial = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    load_dofs_local_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_axial)
    assert np.isclose(load_dofs_local_axial[0], -E * A)
    assert np.isclose(load_dofs_local_axial[6], E * A)
    u_dofs_global_shear = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    load_dofs_local_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_shear)
    assert np.isclose(load_dofs_local_shear[1], 12 * E * ele_info['I_z'])
    assert np.isclose(load_dofs_local_shear[5], -6 * E * ele_info['I_z'])
    assert np.isclose(load_dofs_local_shear[7], -12 * E * ele_info['I_z'])
    assert np.isclose(load_dofs_local_shear[11], -6 * E * ele_info['I_z'])
    u_dofs_global_torsion = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0])
    load_dofs_local_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_torsion)
    assert np.isclose(load_dofs_local_torsion[3], -G * J)
    assert np.isclose(load_dofs_local_torsion[9], G * J)

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a combined displacement state (ua + ub) equals the sum of the individual responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    u_dofs_global_a = np.random.rand(12)
    u_dofs_global_b = np.random.rand(12)
    load_dofs_local_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_a)
    load_dofs_local_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_b)
    load_dofs_local_sum = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_a + u_dofs_global_b)
    assert np.allclose(load_dofs_local_a + load_dofs_local_b, load_dofs_local_sum)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance: If we rotate the entire configuration (coords, displacements, and local_z) by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    u_dofs_global = np.random.rand(12)
    R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    load_dofs_local_original = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    rotated_coords_i = np.dot(R, np.array([xi, yi, zi]))
    rotated_coords_j = np.dot(R, np.array([xj, yj, zj]))
    rotated_u_dofs_global = np.zeros(12)
    for i in range(4):
        rotated_u_dofs_global[i * 3:(i + 1) * 3] = np.dot(R, u_dofs_global[i * 3:(i + 1) * 3])
    rotated_local_z = np.dot(R, ele_info['local_z'])
    ele_info_rotated = ele_info.copy()
    ele_info_rotated['local_z'] = rotated_local_z
    load_dofs_local_rotated = fcn(ele_info_rotated, *rotated_coords_i, *rotated_coords_j, rotated_u_dofs_global)
    assert np.allclose(load_dofs_local_original, load_dofs_local_rotated)