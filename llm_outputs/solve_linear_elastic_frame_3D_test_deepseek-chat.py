def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    n_elements = 10
    element_length = L / n_elements
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = I_z = 1e-05
    J = 2e-05
    direction = np.array([1, 1, 1]) / np.sqrt(3)
    node_coords = np.array([i * element_length * direction for i in range(n_elements + 1)])
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    tip_node = n_elements
    F_mag = 1000.0
    perp_force = F_mag * np.cross(direction, np.array([1, 0, 0]))
    perp_force = perp_force / np.linalg.norm(perp_force) * F_mag
    nodal_loads = {tip_node: [*perp_force, 0, 0, 0]}
    I = I_z
    analytical_deflection = F_mag * L ** 3 / (3 * E * I)
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_disp = u[6 * tip_node:6 * tip_node + 3]
    computed_deflection = np.linalg.norm(tip_disp)
    assert pytest.approx(computed_deflection, rel=0.01) == analytical_deflection

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]])
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = I_z = 1e-05
    J = 2e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 3, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 0, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 4, 'node_j': 5, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 5, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u, 0)
    assert np.allclose(r, 0)
    nodal_loads = {1: [1000, 2000, 3000, 4000, 5000, 6000], 2: [-1000, 0, 1000, 0, -1000, 0], 4: [0, 0, 5000, 0, 0, 0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u1, 0)
    assert not np.allclose(r1, 0)
    doubled_loads = {k: [2 * x for x in v] for (k, v) in nodal_loads.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, doubled_loads)
    assert np.allclose(2 * u1, u2)
    assert np.allclose(2 * r1, r2)
    negated_loads = {k: [-x for x in v] for (k, v) in nodal_loads.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, negated_loads)
    assert np.allclose(-u1, u3)
    assert np.allclose(-r1, r3)
    total_applied_force = np.zeros(3)
    total_applied_moment = np.zeros(3)
    for (node, loads) in nodal_loads.items():
        force = np.array(loads[:3])
        moment = np.array(loads[3:6])
        total_applied_force += force
        total_applied_moment += moment + np.cross(node_coords[node], force)
    reaction_force = r1[:6]
    fixed_node_force = reaction_force[:3]
    fixed_node_moment = reaction_force[3:6]
    assert np.allclose(total_applied_force, -fixed_node_force)
    assert np.allclose(total_applied_moment, -fixed_node_moment)