def test_rigid_body_motion_zero_loads(fcn):
    """Verify that rigid-body translation produces zero internal forces/moments."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    translation = [1.0, 2.0, -0.5]
    u_dofs_global = [*translation, 0, 0, 0, *translation, 0, 0, 0]
    loads = fcn(ele_info, 0, 0, 0, 3, 0, 0, u_dofs_global)
    assert all((abs(load) < 1e-10 for load in loads))

def test_unit_responses_axial_shear_torsion(fcn):
    """Test unit axial, shear, and torsional responses."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    L = 2.0
    u_axial = [0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    loads_axial = fcn(ele_info, 0, 0, 0, L, 0, 0, u_axial)
    assert abs(loads_axial[0] + loads_axial[6]) < 1e-10
    u_shear = [0, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    loads_shear = fcn(ele_info, 0, 0, 0, L, 0, 0, u_shear)
    assert abs(loads_shear[1] + loads_shear[7]) < 1e-10
    u_torsion = [0, 0, 0, 0.001, 0, 0, 0, 0, 0, 0, 0, 0]
    loads_torsion = fcn(ele_info, 0, 0, 0, L, 0, 0, u_torsion)
    assert abs(loads_torsion[3] + loads_torsion[9]) < 1e-10

def test_superposition_linearity(fcn):
    """Verify superposition principle for combined displacements."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    u1 = [0.001, 0.002, 0, 0.001, 0, 0, 0, 0, 0, 0, 0, 0]
    u2 = [0, 0, 0.001, 0, 0.001, 0.001, 0, 0, 0, 0, 0, 0]
    u_combined = [a + b for (a, b) in zip(u1, u2)]
    loads1 = fcn(ele_info, 0, 0, 0, 2, 0, 0, u1)
    loads2 = fcn(ele_info, 0, 0, 0, 2, 0, 0, u2)
    loads_combined = fcn(ele_info, 0, 0, 0, 2, 0, 0, u_combined)
    loads_sum = [f1 + f2 for (f1, f2) in zip(loads1, loads2)]
    assert all((abs(fc - fs) < 1e-10 for (fc, fs) in zip(loads_combined, loads_sum)))

def test_coordinate_invariance_global_rotation(fcn):
    """Verify invariance of local loads under global coordinate rotation."""
    import numpy as np
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    u_orig = [0.001, 0.002, -0.001, 0.001, 0.002, 0, 0, 0, 0, 0, 0, 0]
    loads_orig = fcn(ele_info, 0, 0, 0, 2, 0, 0, u_orig)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    (xi, yi, zi) = R @ np.array([0, 0, 0])
    (xj, yj, zj) = R @ np.array([2, 0, 0])
    R_dof = np.block([[R, np.zeros((3, 9))], [np.zeros((3, 3)), R, np.zeros((3, 6))], [np.zeros((3, 6)), R, np.zeros((3, 3))], [np.zeros((3, 9)), R]])
    u_rotated = R_dof @ np.array(u_orig)
    loads_rotated = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_rotated)
    assert all((abs(lo - lr) < 1e-10 for (lo, lr) in zip(loads_orig, loads_rotated)))