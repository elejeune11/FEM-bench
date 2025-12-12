def test_rigid_body_motion_zero_loads(fcn):
    """
    Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector.
    """
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003}
    (xi, yi, zi) = (1.0, 1.0, 1.0)
    (xj, yj, zj) = (4.0, 5.0, 1.0)
    (dx, dy, dz) = (0.5, -0.2, 1.0)
    u_dofs_global = np.array([dx, dy, dz, 0.0, 0.0, 0.0, dx, dy, dz, 0.0, 0.0, 0.0])
    local_loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(local_loads, 0.0, atol=1e-09)

def test_unit_responses_axial_shear_torsion(fcn):
    """
    Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation
    """
    E = 200000000000.0
    nu = 0.25
    G = E / (2 * (1 + nu))
    A = 0.02
    Iy = 1e-05
    Iz = 2e-05
    J = 5e-05
    L = 2.0
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    delta = 0.0001
    u_axial = np.zeros(12)
    u_axial[6] = delta
    loads_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    axial_stiffness = E * A / L
    assert np.isclose(loads_axial[0], -axial_stiffness * delta)
    assert np.isclose(loads_axial[6], axial_stiffness * delta)
    u_shear = np.zeros(12)
    u_shear[7] = delta
    loads_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear)
    k_force = 12 * E * Iz / L ** 3
    k_moment = 6 * E * Iz / L ** 2
    assert np.isclose(loads_shear[1], -k_force * delta)
    assert np.isclose(loads_shear[5], -k_moment * delta)
    assert np.isclose(loads_shear[7], k_force * delta)
    assert np.isclose(loads_shear[11], -k_moment * delta)
    alpha = 0.001
    u_torsion = np.zeros(12)
    u_torsion[9] = alpha
    loads_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    torsional_stiffness = G * J / L
    assert np.isclose(loads_torsion[3], -torsional_stiffness * alpha)
    assert np.isclose(loads_torsion[9], torsional_stiffness * alpha)

def test_superposition_linearity(fcn):
    """
    Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds.
    """
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (3.0, 4.0, 0.0)
    rng = np.random.default_rng(42)
    ua = rng.random(12) * 0.01
    ub = rng.random(12) * 0.01
    f_ua = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    f_ub = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    f_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
    assert np.allclose(f_combined, f_ua + f_ub, atol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """
    Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged.
    """
    L = 3.0
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 5e-05, 'J': 2e-05, 'local_z': [0, 0, 1]}
    (xi1, yi1, zi1) = (0.0, 0.0, 0.0)
    (xj1, yj1, zj1) = (L, 0.0, 0.0)
    (dx, dy) = (0.001, 0.002)
    u_1 = np.zeros(12)
    u_1[6] = dx
    u_1[7] = dy
    local_loads_1 = fcn(ele_info, xi1, yi1, zi1, xj1, yj1, zj1, u_1)
    (xi2, yi2, zi2) = (0.0, 0.0, 0.0)
    (xj2, yj2, zj2) = (0.0, L, 0.0)
    u_2 = np.zeros(12)
    u_2[6] = -dy
    u_2[7] = dx
    ele_info_2 = ele_info.copy()
    ele_info_2['local_z'] = [0, 0, 1]
    local_loads_2 = fcn(ele_info_2, xi2, yi2, zi2, xj2, yj2, zj2, u_2)
    assert np.allclose(local_loads_1, local_loads_2, atol=1e-10)