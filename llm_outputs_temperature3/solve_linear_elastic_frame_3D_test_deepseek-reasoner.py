def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L_total = 10.0
    n_elements = 10
    L_element = L_total / n_elements
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.33e-06
    I_z = 8.33e-06
    J = 1.67e-05
    beam_dir = np.array([1, 1, 1])
    beam_dir = beam_dir / np.linalg.norm(beam_dir)
    node_coords = []
    for i in range(n_elements + 1):
        pos = i * L_element * beam_dir
        node_coords.append(pos)
    node_coords = np.array(node_coords)
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_tip = 1000.0
    if abs(beam_dir[0]) > 1e-06:
        perp_dir = np.array([-beam_dir[1], beam_dir[0], 0])
    else:
        perp_dir = np.array([0, -beam_dir[2], beam_dir[1]])
    perp_dir = perp_dir / np.linalg.norm(perp_dir)
    tip_load = F_tip * perp_dir
    nodal_loads = {n_elements: [tip_load[0], tip_load[1], tip_load[2], 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    I = I_y
    analytical_deflection_magnitude = F_tip * L_total ** 3 / (3 * E * I)
    tip_node_dofs = n_elements * 6
    computed_displacement = u[tip_node_dofs:tip_node_dofs + 3]
    computed_deflection_magnitude = np.linalg.norm(computed_displacement)
    dot_product = np.dot(computed_displacement / computed_deflection_magnitude, perp_dir)
    tolerance = 0.02
    assert abs(computed_deflection_magnitude - analytical_deflection_magnitude) / analytical_deflection_magnitude < tolerance
    assert abs(dot_product - 1.0) < 0.01

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0, 0, 0], [5, 0, 0], [5, 3, 0], [0, 3, 0], [5, 0, 2], [0, 3, 2]])
    E = 200000000000.0
    nu = 0.3
    A = 0.05
    I_y = 0.0001
    I_z = 0.0001
    J = 0.0002
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 3, 'node_j': 5, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 4, 'node_j': 5, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u1, 0.0, atol=1e-10)
    fixed_dofs = list(range(6))
    assert np.allclose(r1[fixed_dofs], 0.0, atol=1e-10)
    nodal_loads_mixed = {2: [10.0, -5.0, 3.0, 2.0, -1.0, 0.5], 4: [0.0, 8.0, -2.0, 0.0, 0.0, 1.0], 5: [-3.0, 0.0, 4.0, -0.5, 0.3, 0.0]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_mixed)
    assert not np.allclose(u2, 0.0, atol=1e-10)
    assert not np.allclose(r2[fixed_dofs], 0.0, atol=1e-10)
    nodal_loads_double = {}
    for (node, loads) in nodal_loads_mixed.items():
        nodal_loads_double[node] = [2 * x for x in loads]
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u3, 2 * u2, rtol=1e-08)
    assert np.allclose(r3, 2 * r2, rtol=1e-08)
    nodal_loads_negated = {}
    for (node, loads) in nodal_loads_mixed.items():
        nodal_loads_negated[node] = [-x for x in loads]
    (u4, r4) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negated)
    assert np.allclose(u4, -u2, rtol=1e-08)
    assert np.allclose(r4, -r2, rtol=1e-08)
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    for (node, loads) in nodal_loads_mixed.items():
        force = np.array(loads[0:3])
        moment = np.array(loads[3:6])
        position = node_coords[node]
        total_force += force
        total_moment += moment + np.cross(position, force)
    reaction_force = np.array(r2[0:3])
    reaction_moment = np.array(r2[3:6])
    support_position = node_coords[0]
    total_force += reaction_force
    total_moment += reaction_moment + np.cross(support_position, reaction_force)
    assert np.allclose(total_force, 0.0, atol=1e-08)
    assert np.allclose(total_moment, 0.0, atol=1e-08)