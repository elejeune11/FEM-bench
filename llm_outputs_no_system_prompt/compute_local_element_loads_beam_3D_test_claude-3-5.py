def test_rigid_body_motion_zero_loads(fcn):
    """Verify that rigid-body translation produces zero internal forces/moments."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    translation = [1.0, 2.0, -0.5]
    u_rigid = translation * 2 + [0.0] * 6
    loads = fcn(ele_info, 0, 0, 0, 1, 0, 0, u_rigid)
    assert numpy.allclose(loads, numpy.zeros(12), atol=1e-10)

def test_unit_responses_axial_shear_torsion(fcn):
    """Test unit axial, shear, and torsional responses independently."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    L = 1.0
    u_axial = [0.0] * 12
    u_axial[6] = 0.001
    loads_axial = fcn(ele_info, 0, 0, 0, L, 0, 0, u_axial)
    assert numpy.isclose(loads_axial[0], -ele_info['E'] * ele_info['A'] * 0.001 / L)
    assert numpy.isclose(loads_axial[6], ele_info['E'] * ele_info['A'] * 0.001 / L)
    u_shear = [0.0] * 12
    u_shear[7] = 0.001
    loads_shear = fcn(ele_info, 0, 0, 0, L, 0, 0, u_shear)
    assert numpy.isclose(loads_shear[1], -12 * ele_info['E'] * ele_info['I_z'] * 0.001 / L ** 3)
    u_torsion = [0.0] * 12
    u_torsion[9] = 0.001
    loads_torsion = fcn(ele_info, 0, 0, 0, L, 0, 0, u_torsion)
    assert numpy.isclose(loads_torsion[3], -ele_info['G'] * ele_info['J'] * 0.001 / L)

def test_superposition_linearity(fcn):
    """Verify that superposition principle holds for combined displacements."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    u1 = numpy.array([0.001, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    u2 = numpy.array([0, 0, 0.001, 0, 0, 0.002, 0, 0, 0, 0, 0, 0])
    loads1 = fcn(ele_info, 0, 0, 0, 1, 0, 0, u1)
    loads2 = fcn(ele_info, 0, 0, 0, 1, 0, 0, u2)
    loads_combined = fcn(ele_info, 0, 0, 0, 1, 0, 0, u1 + u2)
    assert numpy.allclose(loads_combined, loads1 + loads2, rtol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """Verify that global rotation of the system preserves local internal forces."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    u_orig = numpy.array([0.001, 0.002, -0.001, 0.001, 0, 0, 0, 0, 0, 0, 0.002, 0])
    loads_orig = fcn(ele_info, 0, 0, 0, 1, 0, 0, u_orig)
    c = numpy.cos(numpy.pi / 4)
    s = numpy.sin(numpy.pi / 4)
    R = numpy.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    coords_i = numpy.array([0, 0, 0])
    coords_j = numpy.array([1, 0, 0])
    coords_i_rot = R @ coords_i
    coords_j_rot = R @ coords_j
    R_block = numpy.block([[R, numpy.zeros((3, 3))], [numpy.zeros((3, 3)), R]])
    R_full = numpy.block([[R_block, numpy.zeros((6, 6))], [numpy.zeros((6, 6)), R_block]])
    u_rot = R_full @ u_orig
    loads_rot = fcn(ele_info, coords_i_rot[0], coords_i_rot[1], coords_i_rot[2], coords_j_rot[0], coords_j_rot[1], coords_j_rot[2], u_rot)
    assert numpy.allclose(loads_rot, loads_orig, rtol=1e-10)