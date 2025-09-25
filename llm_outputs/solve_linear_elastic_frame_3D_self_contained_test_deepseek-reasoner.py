def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    n_elements = 10
    L_total = 10.0
    node_coords = []
    for i in range(n_elements + 1):
        t = i / n_elements
        node_coords.append([t * L_total, t * L_total, t * L_total])
    node_coords = np.array(node_coords)
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-05
    I_z = 1e-05
    J = 2e-05
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_magnitude = 1000.0
    force_dir = np.array([1, -1, 0])
    force_dir = force_dir / np.linalg.norm(force_dir)
    force_vector = F_magnitude * force_dir
    nodal_loads = {n_elements: [force_vector[0], force_vector[1], force_vector[2], 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    L = L_total
    I_effective = I_z
    F_perp = F_magnitude
    expected_deflection_magnitude = F_perp * L ** 3 / (3 * E * I_effective)
    tip_disp_index = 6 * n_elements
    tip_displacement = u[tip_disp_index:tip_disp_index + 3]
    actual_deflection_magnitude = np.linalg.norm(tip_displacement)
    assert abs(actual_deflection_magnitude - expected_deflection_magnitude) / expected_deflection_magnitude < 0.01

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions."""
    node_coords = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0], [0, 5, 0], [0, 0, 5], [5, 0, 5], [5, 5, 5], [0, 5, 5]])
    elements = []
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-05
    I_z = 1e-05
    J = 2e-05
    elements.extend([{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 3, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}])
    elements.extend([{'node_i': 4, 'node_j': 5, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 5, 'node_j': 6, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 6, 'node_j': 7, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 7, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}])
    elements.extend([{'node_i': 0, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 5, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 2, 'node_j': 6, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 3, 'node_j': 7, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}])
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u1, 0.0)
    assert np.allclose(r1[6:], 0.0)
    nodal_loads = {5: [1000, 500, -200, 50, -30, 10], 6: [-500, 300, 100, -20, 15, -5]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u2, 0.0)
    assert not np.allclose(r2[:6], 0.0)
    nodal_loads_double = {5: [2000, 1000, -400, 100, -60, 20], 6: [-1000, 600, 200, -40, 30, -10]}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u3, 2 * u2, rtol=1e-10)
    assert np.allclose(r3, 2 * r2, rtol=1e-10)
    nodal_loads_negated = {5: [-1000, -500, 200, -50, 30, -10], 6: [500, -300, -100, 20, -15, 5]}
    (u4, r4) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negated)
    assert np.allclose(u4, -u2, rtol=1e-10)
    assert np.allclose(r4, -r2, rtol=1e-10)
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    for (node, loads) in nodal_loads.items():
        total_force += np.array(loads[:3])
        node_pos = node_coords[node]
        total_moment += np.cross(node_pos, np.array(loads[:3])) + np.array(loads[3:6])
    for (node, bc) in boundary_conditions.items():
        node_start = 6 * node
        reaction_forces = r2[node_start:node_start + 6]
        total_force += np.array(reaction_forces[:3])
        node_pos = node_coords[node]
        total_moment += np.cross(node_pos, np.array(reaction_forces[:3])) + np.array(reaction_forces[3:6])
    assert np.allclose(total_force, 0.0, atol=1e-10)
    assert np.allclose(total_moment, 0.0, atol=1e-10)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError when the structure is improperly constrained."""
    node_coords = np.array([[0, 0, 0], [5, 0, 0]])
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-05
    I_z = 1e-05
    J = 2e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}]
    boundary_conditions = {}
    nodal_loads = {1: [1000, 0, 0, 0, 0, 0]}
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)