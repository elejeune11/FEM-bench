def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 210.0, 'nu': 0.3, 'A': 1.0, 'I_y': 0.8, 'I_z': 1.2, 'J': 0.5, 'local_z': [0.0, 0.0, 1.0]}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_dofs_global = np.array([3.2, -1.7, 0.9, 0.0, 0.0, 0.0, 3.2, -1.7, 0.9, 0.0, 0.0, 0.0], dtype=float)
    load_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert isinstance(load_local, np.ndarray)
    assert load_local.shape == (12,)
    assert np.allclose(load_local, np.zeros(12), atol=1e-12, rtol=0.0)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
      (1) Axial unit extension
      (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
      (3) Unit torsional rotation"""
    ele_info = {'E': 1.0, 'nu': 0.0, 'A': 1.0, 'I_y': 1.0, 'I_z': 1.0, 'J': 1.0, 'local_z': [0.0, 0.0, 1.0]}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_axial = np.zeros(12)
    u_axial[0] = 0.0
    u_axial[6] = 1.0
    f_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    expected_axial = np.zeros(12)
    expected_axial[0] = -1.0
    expected_axial[6] = +1.0
    assert np.allclose(f_axial, expected_axial, atol=1e-12, rtol=0.0)
    u_shear = np.zeros(12)
    u_shear[1] = 0.0
    u_shear[7] = 1.0
    f_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear)
    expected_shear = np.zeros(12)
    expected_shear[1] = -12.0
    expected_shear[5] = -6.0
    expected_shear[7] = +12.0
    expected_shear[11] = -6.0
    assert np.allclose(f_shear, expected_shear, atol=1e-12, rtol=0.0)
    u_torsion = np.zeros(12)
    u_torsion[3] = 0.0
    u_torsion[9] = 1.0
    f_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    expected_torsion = np.zeros(12)
    expected_torsion[3] = -0.5
    expected_torsion[9] = +0.5
    assert np.allclose(f_torsion, expected_torsion, atol=1e-12, rtol=0.0)

def test_superposition_linearity(fcn):
    """Verify linearity: f(ua + ub) == f(ua) + f(ub) for an arbitrary 3D element and displacement states."""
    ele_info = {'E': 2.0, 'nu': 0.25, 'A': 3.0, 'I_y': 5.0, 'I_z': 7.0, 'J': 11.0, 'local_z': [0.1, 0.9, 0.2]}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.5, 1.0, 0.5)
    ua = np.array([0.1, -0.2, 0.3, 0.05, -0.04, 0.03, -0.1, 0.2, -0.3, -0.02, 0.01, -0.03], dtype=float)
    ub = np.array([0.07, 0.08, -0.06, -0.01, 0.02, 0.05, 0.09, -0.11, 0.13, 0.06, -0.04, 0.02], dtype=float)
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    fab = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
    assert fa.shape == (12,) and fb.shape == (12,) and (fab.shape == (12,))
    assert np.allclose(fab, fa + fb, rtol=1e-12, atol=1e-12)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance: rotating coordinates, displacements, and local_z by a rigid rotation keeps the local internal end-load vector unchanged."""

    def Rx(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=float)

    def Ry(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=float)

    def Rz(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=float)
    R = Rz(np.deg2rad(30.0)) @ Ry(np.deg2rad(-25.0)) @ Rx(np.deg2rad(15.0))
    ele_info = {'E': 70.0, 'nu': 0.33, 'A': 0.8, 'I_y': 0.06, 'I_z': 0.09, 'J': 0.05, 'local_z': np.array([0.2, 0.7, -0.1], dtype=float)}
    ri = np.array([-1.0, 2.0, -3.0], dtype=float)
    rj = np.array([3.0, 5.0, 1.0], dtype=float)
    u1 = np.array([1.2, -0.8, 0.5], dtype=float)
    th1 = np.array([0.05, -0.03, 0.02], dtype=float)
    u2 = np.array([-0.4, 1.1, -0.6], dtype=float)
    th2 = np.array([-0.02, 0.04, -0.07], dtype=float)
    u_global = np.zeros(12, dtype=float)
    u_global[0:3] = u1
    u_global[3:6] = th1
    u_global[6:9] = u2
    u_global[9:12] = th2
    f_local_ref = fcn(ele_info, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], u_global)
    ri_rot = R @ ri
    rj_rot = R @ rj
    z_rot = R @ ele_info['local_z']
    u_global_rot = np.zeros(12, dtype=float)
    u_global_rot[0:3] = R @ u1
    u_global_rot[3:6] = R @ th1
    u_global_rot[6:9] = R @ u2
    u_global_rot[9:12] = R @ th2
    ele_info_rot = dict(ele_info)
    ele_info_rot['local_z'] = z_rot
    f_local_rot = fcn(ele_info_rot, ri_rot[0], ri_rot[1], ri_rot[2], rj_rot[0], rj_rot[1], rj_rot[2], u_global_rot)
    assert f_local_ref.shape == (12,)
    assert f_local_rot.shape == (12,)
    assert np.allclose(f_local_ref, f_local_rot, rtol=1e-12, atol=1e-12)