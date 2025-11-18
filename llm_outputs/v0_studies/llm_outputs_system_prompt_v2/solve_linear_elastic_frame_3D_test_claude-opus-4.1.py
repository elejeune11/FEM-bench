def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L_total = 10.0
    n_elements = 10
    L_elem = L_total / n_elements
    direction = np.array([1.0, 1.0, 1.0])
    direction = direction / np.linalg.norm(direction)
    node_coords = np.array([i * L_elem * direction for i in range(n_elements + 1)])
    E = 200000000000.0
    nu = 0.3
    d = 0.1
    A = np.pi * d ** 2 / 4
    I = np.pi * d ** 4 / 64
    G = E / (2 * (1 + nu))
    J = 2 * I
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F = 1000.0
    force_perp = np.array([1.0, -1.0, 0.0])
    force_perp = force_perp - np.dot(force_perp, direction) * direction
    force_perp = F * force_perp / np.linalg.norm(force_perp)
    nodal_loads = {n_elements: [force_perp[0], force_perp[1], force_perp[2], 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_disp = u[-6:-3]
    disp_magnitude = np.linalg.norm(tip_disp)
    delta_analytical = F * L_total ** 3 / (3 * E * I)
    assert np.abs(disp_magnitude - delta_analytical) / delta_analytical < 0.01
    total_reaction_force = r[0:3]
    total_reaction_moment = r[3:6]
    assert np.allclose(total_reaction_force + force_perp, 0, atol=1e-06)
    moment_arm = node_coords[-1] - node_coords[0]
    expected_moment = np.cross(moment_arm, force_perp)
    assert np.allclose(total_reaction_moment + expected_moment, 0, atol=1e-06)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 3.0, 0.0], [0.0, 3.0, 0.0], [1.0, 1.5, 2.0], [3.0, 1.5, 1.0]])
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.33e-06
    I_z = 8.33e-06
    J = 1.66e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 3, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 0, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 2, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 5, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 4, 'node_j': 5, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 3: [1, 1, 1, 0, 0, 0]}
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0, atol=1e-10)
    assert np.allclose(r_zero, 0, atol=1e-10)
    nodal_loads_base = {1: [1000.0, -500.0, 200.0, 50.0, -30.0, 20.0], 2: [-800.0, 600.0, -300.0, -40.0, 25.0, -15.0], 4: [500.0, 300.0, -1000.0, 30.0, -20.0, 10.0], 5: [-200.0, -400.0, 800.0, -25.0, 15.0, -10.0]}
    (u_base, r_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads_base)
    assert not np.allclose(u_base, 0, atol=1e-10)
    assert not np.allclose(r_base, 0, atol=1e-10)
    nodal_loads_double = {k: [2 * v_i for v_i in v] for (k, v) in nodal_loads_base.items()}
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u_double, 2 * u_base, rtol=1e-10)
    assert np.allclose(r_double, 2 * r_base, rtol=1e-10)
    nodal_loads_neg = {k: [-v_i for v_i in v] for (k, v) in nodal_loads_base.items()}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u_neg, -u_base, rtol=1e-10)
    assert np.allclose(r_neg, -r_base, rtol=1e-10)
    total_applied_force = np.zeros(3)
    total_applied_moment = np.zeros(3)
    for (node_idx, loads) in nodal_loads_base.items():
        total_applied_force += np.array(loads[:3])
        total_applied_moment += np.array(loads[3:6])
        total_applied_moment += np.cross(node_coords[node_idx], loads[:3])
    total_reaction_force = np.zeros(3)
    total_reaction_moment = np.zeros(3)
    for node_idx in range(len(node_coords)):
        r_node = r_base[6 * node_idx:6 * (node_idx + 1)]
        total_reaction_force += r_node[:3]
        total_reaction_moment += r_node[3:6]
        total_reaction_moment += np.cross(node_coords[node_idx], r_node[:3])
    assert np.allclose(total_applied_force + total_reaction_force, 0, atol=1e-06)
    assert np.allclose(total_applied_moment + total_reaction_moment, 0, atol=1e-06)