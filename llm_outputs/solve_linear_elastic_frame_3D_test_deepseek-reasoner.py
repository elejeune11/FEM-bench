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
    node_coords = []
    for i in range(n_elements + 1):
        t = i * L / n_elements
        node_coords.append([t / np.sqrt(3), t / np.sqrt(3), t / np.sqrt(3)])
    node_coords = np.array(node_coords)
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    tip_node = n_elements
    force_magnitude = 1000
    perp_direction = np.array([1, -1, 0])
    perp_direction = perp_direction / np.linalg.norm(perp_direction)
    force_vector = force_magnitude * perp_direction
    nodal_loads = {tip_node: [force_vector[0], force_vector[1], 0, 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    analytical_deflection = force_magnitude * L ** 3 / (3 * E * I_z)
    tip_disp = u[tip_node * 6:tip_node * 6 + 3]
    tip_disp_magnitude = np.linalg.norm(tip_disp)
    assert np.abs(tip_disp_magnitude - analytical_deflection) / analytical_deflection < 0.01
    disp_direction = tip_disp / tip_disp_magnitude
    expected_direction = perp_direction
    dot_product = np.dot(disp_direction, expected_direction)
    assert np.abs(dot_product - 1.0) < 0.01

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions."""
    node_coords = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0], [5, 5, 5]])
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.33e-06
    I_z = 8.33e-06
    J = 1.67e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u1, 0, atol=1e-10)
    assert np.allclose(r1[np.array([0, 1, 2, 3, 4, 5])], 0, atol=1e-10)
    nodal_loads = {1: [1000, 0, 0, 0, 0, 0], 2: [0, 2000, 0, 0, 0, 0], 3: [0, 0, 3000, 0, 500, 0]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u2, 0, atol=1e-10)
    assert not np.allclose(r2[np.array([0, 1, 2, 3, 4, 5])], 0, atol=1e-10)
    nodal_loads_double = {1: [2000, 0, 0, 0, 0, 0], 2: [0, 4000, 0, 0, 0, 0], 3: [0, 0, 6000, 0, 1000, 0]}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u3, 2 * u2, rtol=1e-08)
    assert np.allclose(r3, 2 * r2, rtol=1e-08)
    nodal_loads_negated = {1: [-1000, 0, 0, 0, 0, 0], 2: [0, -2000, 0, 0, 0, 0], 3: [0, 0, -3000, 0, -500, 0]}
    (u4, r4) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negated)
    assert np.allclose(u4, -u2, rtol=1e-08)
    assert np.allclose(r4, -r2, rtol=1e-08)
    n_nodes = len(node_coords)
    applied_forces = np.zeros(3)
    applied_moments = np.zeros(3)
    for (node, loads) in nodal_loads.items():
        (fx, fy, fz, mx, my, mz) = loads
        applied_forces += np.array([fx, fy, fz])
        r = node_coords[node] - node_coords[0]
        force = np.array([fx, fy, fz])
        applied_moments += np.cross(r, force) + np.array([mx, my, mz])
    reaction_forces = np.zeros(3)
    reaction_moments = np.zeros(3)
    for i in range(n_nodes):
        start_idx = i * 6
        (fx, fy, fz, mx, my, mz) = r2[start_idx:start_idx + 6]
        reaction_forces += np.array([fx, fy, fz])
        if i > 0:
            r = node_coords[i] - node_coords[0]
            force = np.array([fx, fy, fz])
            reaction_moments += np.cross(r, force) + np.array([mx, my, mz])
    total_force = applied_forces + reaction_forces
    assert np.allclose(total_force, 0, atol=1e-08)
    total_moment = applied_moments + reaction_moments
    assert np.allclose(total_moment, 0, atol=1e-08)