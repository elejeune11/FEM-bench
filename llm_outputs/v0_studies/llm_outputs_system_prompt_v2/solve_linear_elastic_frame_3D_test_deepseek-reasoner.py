def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.33e-06
    I_z = 8.33e-06
    J = 1.67e-05
    n_elements = 10
    n_nodes = n_elements + 1
    node_coords = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        t = i * L / n_elements / np.sqrt(3)
        node_coords[i] = [t, t, t]
    elements = []
    for i in range(n_elements):
        element = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}
        elements.append(element)
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    tip_node = n_nodes - 1
    force_magnitude = 1000.0
    force_dir = np.array([1, -1, 0])
    force_dir = force_dir / np.linalg.norm(force_dir)
    force_vector = force_magnitude * force_dir
    nodal_loads = {tip_node: [force_vector[0], force_vector[1], force_vector[2], 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    L_actual = np.linalg.norm(node_coords[-1] - node_coords[0])
    analytical_deflection = force_magnitude * L_actual ** 3 / (3 * E * I_z)
    tip_disp = u[tip_node * 6:tip_node * 6 + 3]
    computed_deflection = np.linalg.norm(tip_disp)
    assert np.abs(computed_deflection - analytical_deflection) / analytical_deflection < 0.02
    disp_dir = tip_disp / np.linalg.norm(tip_disp)
    force_dir_normalized = force_dir / np.linalg.norm(force_dir)
    dot_product = np.dot(disp_dir, force_dir_normalized)
    assert np.abs(dot_product - 1.0) < 0.01

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions."""
    node_coords = np.array([[0, 0, 0], [5, 0, 0], [5, 3, 0], [8, 3, 0], [5, 3, 2]])
    E = 200000000000.0
    nu = 0.3
    A = 0.02
    I_y = 1.67e-05
    I_z = 1.67e-05
    J = 3.33e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 2, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u1, 0.0, atol=1e-12)
    assert np.allclose(r1, 0.0, atol=1e-12)
    nodal_loads_original = {3: [1000, -500, 200, 50, -30, 10], 4: [-200, 800, 300, -20, 40, -5]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_original)
    assert not np.allclose(u2[3 * 6:3 * 6 + 6], 0.0, atol=1e-12)
    assert not np.allclose(u2[4 * 6:4 * 6 + 6], 0.0, atol=1e-12)
    assert not np.allclose(r2[0:6], 0.0, atol=1e-12)
    nodal_loads_double = {3: [2000, -1000, 400, 100, -60, 20], 4: [-400, 1600, 600, -40, 80, -10]}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u3, 2.0 * u2, rtol=1e-10)
    assert np.allclose(r3, 2.0 * r2, rtol=1e-10)
    nodal_loads_negated = {3: [-1000, 500, -200, -50, 30, -10], 4: [200, -800, -300, 20, -40, 5]}
    (u4, r4) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negated)
    assert np.allclose(u4, -u2, rtol=1e-10)
    assert np.allclose(r4, -r2, rtol=1e-10)
    total_force_applied = np.zeros(3)
    total_moment_applied = np.zeros(3)
    for (node, loads) in nodal_loads_original.items():
        force = np.array(loads[:3])
        moment = np.array(loads[3:])
        total_force_applied += force
        total_moment_applied += moment
        total_moment_applied += np.cross(node_coords[node], force)
    total_force_reaction = np.array(r2[0:3])
    total_moment_reaction = np.array(r2[3:6])
    assert np.allclose(total_force_applied + total_force_reaction, 0.0, atol=1e-10)
    total_moment = total_moment_applied + total_moment_reaction
    assert np.allclose(total_moment, 0.0, atol=1e-10)