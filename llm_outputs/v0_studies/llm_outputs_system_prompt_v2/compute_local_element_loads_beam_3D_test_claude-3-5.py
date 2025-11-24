def test_rigid_body_motion_zero_loads(fcn):
    """Verify that rigid-body translation produces zero internal forces/moments."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': [0, 0, 1]}
    delta = [0.01, -0.02, 0.015]
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_dofs_global = [delta[0], delta[1], delta[2], 0, 0, 0, delta[0], delta[1], delta[2], 0, 0, 0]
    loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert numpy.allclose(loads, numpy.zeros(12), atol=1e-10)

def test_unit_responses_axial_shear_torsion(fcn):
    """Test unit axial, shear, and torsional responses."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': [0, 0, 1]}
    L = 2.0
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    u_axial = [0.001, 0, 0, 0, 0, 0, 0.002, 0, 0, 0, 0, 0]
    loads_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    F_axial = ele_info['E'] * ele_info['A'] / L * 0.001
    assert numpy.isclose(loads_axial[0], -F_axial)
    assert numpy.isclose(loads_axial[6], F_axial)
    u_shear = [0, 0.001, 0, 0, 0, 0, 0, 0.002, 0, 0, 0, 0]
    loads_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear)
    assert numpy.any(numpy.abs(loads_shear[[1, 5, 7, 11]]) > 0)
    u_torsion = [0, 0, 0, 0.001, 0, 0, 0, 0, 0, 0.002, 0, 0]
    loads_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    T = ele_info['G'] * ele_info['J'] / L * 0.001
    assert numpy.isclose(loads_torsion[3], -T)
    assert numpy.isclose(loads_torsion[9], T)

def test_superposition_linearity(fcn):
    """Verify superposition principle for combined displacements."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u1 = [0.001, 0.002, -0.001, 0.001, 0.002, -0.001, 0.003, -0.001, 0.002, -0.002, 0.001, 0.003]
    u2 = [-0.002, 0.001, 0.002, -0.001, -0.002, 0.001, 0.001, 0.002, -0.001, 0.001, -0.001, -0.002]
    u_combined = numpy.add(u1, u2)
    f1 = fcn(ele_info, xi, yi, zi, xj, yj, zj, u1)
    f2 = fcn(ele_info, xi, yi, zi, xj, yj, zj, u2)
    f_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_combined)
    assert numpy.allclose(f_combined, f1 + f2, rtol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """Verify invariance under global coordinate rotation."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_orig = [0.001, 0.002, -0.001, 0.001, 0.002, -0.001, 0.003, -0.001, 0.002, -0.002, 0.001, 0.003]
    f_orig = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_orig)
    theta = numpy.pi / 4
    R = numpy.array([[numpy.cos(theta), -numpy.sin(theta), 0], [numpy.sin(theta), numpy.cos(theta), 0], [0, 0, 1]])
    p1 = R @ numpy.array([xi, yi, zi])
    p2 = R @ numpy.array([xj, yj, zj])
    R_block = numpy.block([[R, numpy.zeros((3, 3))], [numpy.zeros((3, 3)), R]])
    u_rot = numpy.concatenate([R_block @ numpy.array(u_orig[:6]), R_block @ numpy.array(u_orig[6:])])
    ele_info_rot = ele_info.copy()
    ele_info_rot['local_z'] = R @ numpy.array(ele_info['local_z'])
    f_rot = fcn(ele_info_rot, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], u_rot)
    assert numpy.allclose(f_rot, f_orig, rtol=1e-10)