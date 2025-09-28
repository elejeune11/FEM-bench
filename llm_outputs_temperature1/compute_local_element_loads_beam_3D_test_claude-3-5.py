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
    assert abs(loads_torsion[3]) > 1e-06

def test_superposition_linearity(fcn):
    """Verify superposition principle for combined displacements."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    u1 = [0.001, 0.002, 0, 0.001, 0, 0, 0, 0, 0, 0, 0, 0]
    u2 = [0, 0, 0.003, 0, 0.002, 0, 0.001, 0, 0, 0, 0.001, 0]
    u_combined = [a + b for (a, b) in zip(u1, u2)]
    loads1 = fcn(ele_info, 0, 0, 0, 3, 0, 0, u1)
    loads2 = fcn(ele_info, 0, 0, 0, 3, 0, 0, u2)
    loads_combined = fcn(ele_info, 0, 0, 0, 3, 0, 0, u_combined)
    loads_sum = [a + b for (a, b) in zip(loads1, loads2)]
    assert all((abs(f1 - f2) < 1e-10 for (f1, f2) in zip(loads_combined, loads_sum)))

def test_coordinate_invariance_global_rotation(fcn):
    """Verify invariance under global coordinate rotation."""
    import numpy as np
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    u_orig = [0.001, 0.002, 0.003, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.004, 0.005, 0.006]
    loads_orig = fcn(ele_info, 0, 0, 0, 3, 0, 0, u_orig)
    theta = np.pi / 4
    (c, s) = (np.cos(theta), np.sin(theta))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    (xi, yi, zi) = R @ [0, 0, 0]
    (xj, yj, zj) = R @ [3, 0, 0]
    R_block = np.kron(np.eye(4), R)
    u_rotated = R_block @ u_orig
    loads_rotated = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_rotated)
    assert all((abs(f1 - f2) < 1e-10 for (f1, f2) in zip(loads_orig, loads_rotated)))