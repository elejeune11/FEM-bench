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
    axis_dir = np.array([1, 1, 1])
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    node_coords = []
    for i in range(n_elements + 1):
        pos = i * L_element * axis_dir
        node_coords.append(pos)
    node_coords = np.array(node_coords)
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    tip_node = n_elements
    F_magnitude = 1000.0
    arbitrary_vec = np.array([0, 0, 1]) if not np.allclose(axis_dir, [0, 0, 1]) else np.array([1, 0, 0])
    load_dir = np.cross(axis_dir, arbitrary_vec)
    load_dir = load_dir / np.linalg.norm(load_dir)
    load_vector = F_magnitude * load_dir
    nodal_loads = {tip_node: [load_vector[0], load_vector[1], load_vector[2], 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    analytical_deflection = F_magnitude * L_total ** 3 / (3 * E * I_z)
    tip_disp_index = tip_node * 6
    tip_displacement = u[tip_disp_index:tip_disp_index + 3]
    computed_deflection = np.linalg.norm(tip_displacement)
    tolerance = 0.02 * analytical_deflection
    assert abs(computed_deflection - analytical_deflection) < tolerance

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions."""
    node_coords = np.array([[0, 0, 0], [2, 0, 0], [2, 3, 0], [2, 3, 2], [0, 3, 2]])
    elements = []
    element_props = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.02, 'I_y': 1.67e-05, 'I_z': 1.67e-05, 'J': 3.33e-05, 'local_z': None}
    connections = [(0, 1), (1, 2), (2, 3), (2, 4)]
    for (i, j) in connections:
        elements.append({'node_i': i, 'node_j': j, **element_props})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u1, 0.0, atol=1e-10)
    assert np.allclose(r1[0:6], 0.0, atol=1e-10)
    nodal_loads = {3: [1000, 500, -200, 50, -30, 10], 4: [-500, 800, 300, -20, 40, -5]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u2, 0.0)
    assert not np.allclose(r2, 0.0)
    nodal_loads_double = {3: [2000, 1000, -400, 100, -60, 20], 4: [-1000, 1600, 600, -40, 80, -10]}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u3, 2 * u2, rtol=1e-08)
    assert np.allclose(r3, 2 * r2, rtol=1e-08)
    nodal_loads_negated = {3: [-1000, -500, 200, -50, 30, -10], 4: [500, -800, -300, 20, -40, 5]}
    (u4, r4) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negated)
    assert np.allclose(u4, -u2, rtol=1e-08)
    assert np.allclose(r4, -r2, rtol=1e-08)
    total_applied_force = np.zeros(3)
    total_applied_moment = np.zeros(3)
    for (node_id, loads) in nodal_loads.items():
        force = np.array(loads[0:3])
        moment = np.array(loads[3:6])
        position = node_coords[node_id]
        total_applied_force += force
        total_applied_moment += moment + np.cross(position, force)
    reaction_force = np.array(r2[0:3])
    reaction_moment = np.array(r2[3:6])
    assert np.allclose(total_applied_force + reaction_force, 0.0, atol=1e-08)
    total_moment = total_applied_moment + reaction_moment
    assert np.allclose(total_moment, 0.0, atol=1e-08)