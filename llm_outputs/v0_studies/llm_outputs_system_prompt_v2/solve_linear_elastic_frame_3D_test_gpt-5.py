def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    n_elems = 10
    L = 1.5
    a = np.array([1.0, 1.0, 1.0])
    a = a / np.linalg.norm(a)
    nodes = n_elems + 1
    node_coords = np.zeros((nodes, 3))
    for i in range(nodes):
        s = L * i / n_elems
        node_coords[i, :] = s * a
    E = 210000000000.0
    nu = 0.3
    r = 0.04
    A = np.pi * r * r
    I = np.pi * r ** 4 / 4.0
    J = 2.0 * I
    elements = []
    for i in range(n_elems):
        p_i = node_coords[i]
        p_j = node_coords[i + 1]
        x_dir = p_j - p_i
        x_dir = x_dir / np.linalg.norm(x_dir)
        z_cand = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(x_dir, z_cand)) > 0.99:
            z_cand = np.array([0.0, 1.0, 0.0])
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': z_cand})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    v_perp = np.array([1.0, -1.0, 0.0])
    v_perp = v_perp / np.linalg.norm(v_perp)
    assert abs(np.dot(a, v_perp)) < 1e-12
    P = 5.0
    F = P * v_perp
    nodal_loads = {nodes - 1: [F[0], F[1], F[2], 0.0, 0.0, 0.0]}
    (u, r_vec) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    u_tip = u[6 * (nodes - 1):6 * (nodes - 1) + 3]
    delta_analytical = P * L ** 3 / (3.0 * E * I)
    disp_along_load = float(np.dot(u_tip, v_perp))
    assert np.isclose(disp_along_load, delta_analytical, rtol=0.001, atol=1e-08)
    disp_along_axis = float(np.dot(u_tip, a))
    assert abs(disp_along_axis) <= max(1e-10, 1e-06 * abs(delta_analytical))
    u_orth = u_tip - disp_along_load * v_perp
    assert np.linalg.norm(u_orth) <= max(1e-10, 1e-06 * abs(delta_analytical))

def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.3, 0.2, 0.1], [0.1, 1.1, 0.4], [0.2, 0.3, 1.4], [1.0, 1.0, 0.8]], dtype=float)
    n_nodes = node_coords.shape[0]
    E = 100000000000.0
    nu = 0.33
    A = 0.003
    I = 2e-06
    J = 4e-06
    conns = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3), (2, 4)]
    elements = []
    for (i, j) in conns:
        p_i = node_coords[i]
        p_j = node_coords[j]
        x_dir = p_j - p_i
        x_dir = x_dir / np.linalg.norm(x_dir)
        z_cand = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(x_dir, z_cand)) > 0.99:
            z_cand = np.array([0.0, 1.0, 0.0])
        elements.append({'node_i': i, 'node_j': j, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': z_cand})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads0 = {}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, nodal_loads0)
    assert np.allclose(u0, 0.0)
    assert np.allclose(r0, 0.0)
    F2 = np.array([100.0, -50.0, 30.0], dtype=float)
    M2 = np.array([5.0, -4.0, 3.0], dtype=float)
    F4 = np.array([-70.0, 20.0, 50.0], dtype=float)
    M4 = np.array([0.0, 7.0, -6.0], dtype=float)
    nodal_loads1 = {2: [F2[0], F2[1], F2[2], M2[0], M2[1], M2[2]], 4: [F4[0], F4[1], F4[2], M4[0], M4[1], M4[2]]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads1)
    assert np.linalg.norm(u1) > 0.0
    assert np.linalg.norm(r1) > 0.0
    nodal_loads2 = {k: [2.0 * v for v in vals] for (k, vals) in nodal_loads1.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads2)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-09, atol=1e-12)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-09, atol=1e-12)
    nodal_loads3 = {k: [-1.0 * v for v in vals] for (k, vals) in nodal_loads1.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads3)
    assert np.allclose(u3, -u1, rtol=1e-09, atol=1e-12)
    assert np.allclose(r3, -r1, rtol=1e-09, atol=1e-12)
    total_applied_force = F2 + F4
    reaction_forces = np.zeros(3)
    reaction_moments = np.zeros(3)
    for i in range(n_nodes):
        reaction_forces += r1[6 * i:6 * i + 3]
        reaction_moments += r1[6 * i + 3:6 * i + 6]
    assert np.allclose(reaction_forces, -total_applied_force, rtol=1e-08, atol=1e-08)
    r2_pos = node_coords[2]
    r4_pos = node_coords[4]
    total_external_torque = np.cross(r2_pos, F2) + M2 + np.cross(r4_pos, F4) + M4
    assert np.allclose(reaction_moments, -total_external_torque, rtol=1e-08, atol=1e-08)