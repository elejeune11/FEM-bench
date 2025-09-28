def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    n_elements = 10
    n_nodes = n_elements + 1
    direction = np.array([1, 1, 1]) / np.sqrt(3)
    node_coords = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        node_coords[i] = i * (L / n_elements) * direction
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.333e-06
    I_z = 8.333e-06
    J = 1.667e-05
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F = 1000.0
    nodal_loads = {n_nodes - 1: [0, F, 0, 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    delta_analytical = F * L ** 3 / (3 * E * I_y)
    tip_displacement = u[(n_nodes - 1) * 6:(n_nodes - 1) * 6 + 3]
    tip_deflection_magnitude = np.linalg.norm(tip_displacement)
    assert abs(tip_deflection_magnitude - delta_analytical) / delta_analytical < 0.05
    fixed_displacement = u[0:6]
    assert np.allclose(fixed_displacement, 0, atol=1e-12)
    support_reaction = r[0:6]
    support_force_magnitude = np.linalg.norm(support_reaction[0:3])
    assert abs(support_force_magnitude - F) / F < 1e-06

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [2, 0, 2]])
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.333e-06
    I_z = 8.333e-06
    J = 1.667e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u1, 0, atol=1e-12)
    assert np.allclose(r1, 0, atol=1e-12)
    nodal_loads_mixed = {2: [100, 200, 50, 10, 20, 5], 3: [150, -100, 75, -15, 10, -8]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_mixed)
    free_displacements = np.concatenate([u2[6:12], u2[12:18], u2[18:24]])
    assert np.any(np.abs(free_displacements) > 1e-10)
    support_reactions = r2[0:6]
    assert np.any(np.abs(support_reactions) > 1e-06)
    nodal_loads_double = {2: [200, 400, 100, 20, 40, 10], 3: [300, -200, 150, -30, 20, -16]}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u3, 2 * u2, rtol=1e-10)
    assert np.allclose(r3, 2 * r2, rtol=1e-10)
    nodal_loads_negative = {2: [-100, -200, -50, -10, -20, -5], 3: [-150, 100, -75, 15, -10, 8]}
    (u4, r4) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negative)
    assert np.allclose(u4, -u2, rtol=1e-10)
    assert np.allclose(r4, -r2, rtol=1e-10)
    applied_forces = np.array([100, 200, 50]) + np.array([150, -100, 75])
    applied_moments = np.array([10, 20, 5]) + np.array([-15, 10, -8])
    reaction_forces = r2[0:3]
    reaction_moments = r2[3:6]
    total_forces = applied_forces + reaction_forces
    assert np.allclose(total_forces, 0, atol=1e-06)
    moment_from_forces_node2 = np.cross(node_coords[2], nodal_loads_mixed[2][:3])
    moment_from_forces_node3 = np.cross(node_coords[3], nodal_loads_mixed[3][:3])
    total_applied_moments = applied_moments + moment_from_forces_node2 + moment_from_forces_node3
    total_moments = total_applied_moments + reaction_moments
    assert np.allclose(total_moments, 0, atol=1e-06)