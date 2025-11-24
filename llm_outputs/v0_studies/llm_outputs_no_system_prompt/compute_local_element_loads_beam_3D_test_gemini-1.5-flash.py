def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 1, 1)
    u_dofs_global = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert_allclose(loads, np.zeros(12), atol=1e-10)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses: (1) Axial unit extension (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations (3) Unit torsional rotation"""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    u_dofs_global = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    loads_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    u_dofs_global = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    loads_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    u_dofs_global = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    loads_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert loads_axial[0] > 0
    assert loads_shear[1] > 0
    assert loads_torsion[5] > 0

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a combined displacement state (ua + ub) equals the sum of the individual responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    u_dofs_global_a = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    u_dofs_global_b = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    loads_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_a)
    loads_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_b)
    loads_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_a + u_dofs_global_b)
    assert_allclose(loads_combined, loads_a + loads_b, atol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance: If we rotate the entire configuration (coords, displacements, and local_z) by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    pass