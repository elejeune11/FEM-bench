def test_rigid_body_motion_zero_loads(fcn):
    """Verify that rigid-body translation produces zero internal forces/moments."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': [0, 0, 1]}
    translation = [0.1, -0.2, 0.3]
    u_dofs_global = [*translation, 0, 0, 0, *translation, 0, 0, 0]
    loads = fcn(ele_info, 0, 0, 0, 1, 0, 0, u_dofs_global)
    assert numpy.allclose(loads, numpy.zeros(12), atol=1e-10)

def test_unit_responses_axial_shear_torsion(fcn):
    """Test unit axial, shear, and torsional responses."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': [0, 0, 1]}
    L = 2.0
    u_axial = [0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    loads_axial = fcn(ele_info, 0, 0, 0, L, 0, 0, u_axial)
    F_axial = ele_info['E'] * ele_info['A'] * 0.001 / L
    assert numpy.isclose(loads_axial[0], -F_axial)
    assert numpy.isclose(loads_axial[6], F_axial)
    u_shear = [0, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    loads_shear = fcn(ele_info, 0, 0, 0, L, 0, 0, u_shear)
    assert numpy.any(numpy.abs(loads_shear[[1, 5, 7, 11]]) > 0)
    u_torsion = [0, 0, 0, 0.001, 0, 0, 0, 0, 0, 0, 0, 0]
    loads_torsion = fcn(ele_info, 0, 0, 0, L, 0, 0, u_torsion)
    T = ele_info['G'] * ele_info['J'] * 0.001 / L
    assert numpy.isclose(loads_torsion[3], -T)
    assert numpy.isclose(loads_torsion[9], T)

def test_superposition_linearity(fcn):
    """Verify superposition principle for combined displacements."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': [0, 0, 1]}
    u1 = [0.001, 0.002, -0.001, 0.001, 0, 0, 0, 0, 0, 0, 0, 0]
    u2 = [0, 0, 0, 0, 0.002, 0.001, 0.001, -0.001, 0.002, 0, 0, 0]
    u_combined = numpy.add(u1, u2)
    loads1 = fcn(ele_info, 0, 0, 0, 1, 0, 0, u1)
    loads2 = fcn(ele_info, 0, 0, 0, 1, 0, 0, u2)
    loads_combined = fcn(ele_info, 0, 0, 0, 1, 0, 0, u_combined)
    assert numpy.allclose(loads_combined, numpy.add(loads1, loads2), rtol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """Verify invariance under global coordinate rotation."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': [0, 0, 1]}
    theta = numpy.pi / 4
    R = numpy.array([[numpy.cos(theta), -numpy.sin(theta), 0], [numpy.sin(theta), numpy.cos(theta), 0], [0, 0, 1]])
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (2, 0, 0)
    u_orig = [0.001, 0.002, -0.001, 0.001, 0.002, 0, 0, 0, 0, 0, 0, 0]
    loads_orig = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_orig)
    pi = numpy.array([xi, yi, zi])
    pj = numpy.array([xj, yj, zj])
    pi_rot = R @ pi
    pj_rot = R @ pj
    R_block = numpy.block([[R, numpy.zeros((3, 3))], [numpy.zeros((3, 3)), R]])
    R_full = numpy.block([[R_block, numpy.zeros((6, 6))], [numpy.zeros((6, 6)), R_block]])
    u_rot = R_full @ u_orig
    ele_info_rot = ele_info.copy()
    ele_info_rot['local_z'] = R @ ele_info['local_z']
    loads_rot = fcn(ele_info_rot, *pi_rot, *pj_rot, u_rot)
    assert numpy.allclose(loads_rot, loads_orig, rtol=1e-10)