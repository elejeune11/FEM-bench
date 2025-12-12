def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force
    perpendicular to the beam axis. Verify beam tip deflection with the analytical reference solution
    δ = P*L^3 / (3*E*I) for isotropic bending (I_y = I_z).
    """
    Nseg = 10
    L = 3.0
    axis = np.array([1.0, 1.0, 1.0])
    axis_unit = axis / np.linalg.norm(axis)
    step = L / Nseg
    node_coords = np.array([i * step * axis_unit for i in range(Nseg + 1)], dtype=float)
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I = 8e-06
    J = 2.0 * I
    elements = []
    for i in range(Nseg):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': [0.0, 0.0, 1.0]})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    d = np.array([1.0, -1.0, 0.0], dtype=float)
    assert abs(np.dot(d, axis_unit)) < 1e-12
    d_unit = d / np.linalg.norm(d)
    P = 1000.0
    tip_node = Nseg
    nodal_loads = {tip_node: [P * d_unit[0], P * d_unit[1], P * d_unit[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    delta_expected = P * L ** 3 / (3.0 * E * I)
    u_tip = u[6 * tip_node:6 * tip_node + 3]
    disp_along_load = float(np.dot(u_tip, d_unit))
    assert np.isclose(disp_along_load, delta_expected, rtol=0.0001, atol=1e-08)

def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, -0.2], [2.2, 1.0, 1.3], [1.0, 0.5, 2.0], [0.0, 0.5, 2.3]], dtype=float)
    N = node_coords.shape[0]
    E = 210000000000.0
    nu = 0.3
    A = 0.006
    I = 5e-06
    J = 2.0 * I
    elements = []
    z_global = np.array([0.0, 0.0, 1.0])
    y_global = np.array([0.0, 1.0, 0.0])
    for i in range(N - 1):
        p_i = node_coords[i]
        p_j = node_coords[i + 1]
        axis = p_j - p_i
        axis_unit = axis / np.linalg.norm(axis)
        if abs(np.dot(axis_unit, z_global)) > 0.99:
            local_z = y_global
        else:
            local_z = z_global
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z.tolist()})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    loads_zero = {}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, loads_zero)
    assert np.allclose(u0, 0.0, atol=1e-12)
    assert np.allclose(r0, 0.0, atol=1e-12)
    F3 = np.array([120.0, -200.0, 150.0])
    M3 = np.array([15.0, -10.0, 20.0])
    F5 = np.array([-80.0, 60.0, 90.0])
    M5 = np.array([5.0, 25.0, -15.0])
    loads = {3: [F3[0], F3[1], F3[2], M3[0], M3[1], M3[2]], 5: [F5[0], F5[1], F5[2], M5[0], M5[1], M5[2]]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, loads)
    assert np.linalg.norm(u1) > 0.0
    assert np.linalg.norm(r1) > 0.0
    loads_2x = {3: (2.0 * np.array(loads[3])).tolist(), 5: (2.0 * np.array(loads[5])).tolist()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, loads_2x)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-10, atol=1e-10)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-10, atol=1e-10)
    loads_neg = {3: (-np.array(loads[3])).tolist(), 5: (-np.array(loads[5])).tolist()}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, loads_neg)
    assert np.allclose(u_neg, -u1, rtol=1e-10, atol=1e-10)
    assert np.allclose(r_neg, -r1, rtol=1e-10, atol=1e-10)
    r_support_force = r1[0:3]
    r_support_moment = r1[3:6]
    F_ext = F3 + F5
    p3 = node_coords[3]
    p5 = node_coords[5]
    M_ext = M3 + M5 + np.cross(p3, F3) + np.cross(p5, F5)
    assert np.allclose(r_support_force + F_ext, np.zeros(3), rtol=1e-10, atol=1e-08)
    assert np.allclose(r_support_moment + M_ext, np.zeros(3), rtol=1e-10, atol=1e-08)