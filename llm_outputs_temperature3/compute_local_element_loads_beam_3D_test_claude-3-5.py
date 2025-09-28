def test_rigid_body_motion_zero_loads(fcn):
    """Verify that rigid-body translation produces zero internal forces/moments."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    translation = [1.0, 2.0, -0.5]
    u_dofs_global = [*translation, 0, 0, 0, *translation, 0, 0, 0]
    loads = fcn(ele_info, 0, 0, 0, 3, 0, 0, u_dofs_global)
    assert all((abs(load) < 1e-10 for load in loads))

def test_unit_responses_axial_shear_torsion(fcn):
    """Test axial, shear, and torsional unit responses."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    L = 2.0
    u_axial = [0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    loads_axial = fcn(ele_info, 0, 0, 0, L, 0, 0, u_axial)
    F_axial = ele_info['E'] * ele_info['A'] * 0.001 / L
    assert abs(loads_axial[0] + F_axial) < 1e-06
    assert abs(loads_axial[6] - F_axial) < 1e-06
    u_shear = [0, 0.001, 0, 0, 0, 0, 0, -0.001, 0, 0, 0, 0]
    loads_shear = fcn(ele_info, 0, 0, 0, L, 0, 0, u_shear)
    assert abs(loads_shear[1]) > 1e-06
    u_torsion = [0, 0, 0, 0.001, 0, 0, 0, 0, 0, -0.001, 0, 0]
    loads_torsion = fcn(ele_info, 0, 0, 0, L, 0, 0, u_torsion)
    T = ele_info['G'] * ele_info['J'] * 0.002 / L
    assert abs(abs(loads_torsion[3]) - T) < 1e-06

def test_superposition_linearity(fcn):
    """Verify superposition principle for combined displacements."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    u1 = [0.001, 0.002, 0, 0, 0, 0.001, 0, 0, 0, 0, 0, 0]
    u2 = [0, 0, 0.001, 0.001, 0, 0, 0.002, 0, 0, 0, 0.001, 0]
    u_combined = [a + b for (a, b) in zip(u1, u2)]
    loads1 = fcn(ele_info, 0, 0, 0, 3, 0, 0, u1)
    loads2 = fcn(ele_info, 0, 0, 0, 3, 0, 0, u2)
    loads_combined = fcn(ele_info, 0, 0, 0, 3, 0, 0, u_combined)
    loads_sum = [f1 + f2 for (f1, f2) in zip(loads1, loads2)]
    assert all((abs(fc - fs) < 1e-10 for (fc, fs) in zip(loads_combined, loads_sum)))

def test_coordinate_invariance_global_rotation(fcn):
    """Verify invariance under global coordinate rotation."""
    import numpy as np
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    u_orig = [0.001, 0.002, -0.001, 0.001, 0, 0, 0.002, 0, 0, 0, 0.001, 0]
    loads_orig = fcn(ele_info, 0, 0, 0, 3, 0, 0, u_orig)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    coords_i = R @ np.array([0, 0, 0])
    coords_j = R @ np.array([3, 0, 0])
    R_block = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
    R_full = np.block([[R_block, np.zeros((6, 6))], [np.zeros((6, 6)), R_block]])
    u_rotated = R_full @ np.array(u_orig)
    loads_rotated = fcn(ele_info, coords_i[0], coords_i[1], coords_i[2], coords_j[0], coords_j[1], coords_j[2], u_rotated)
    assert all((abs(lo - lr) < 1e-10 for (lo, lr) in zip(loads_orig, loads_rotated)))