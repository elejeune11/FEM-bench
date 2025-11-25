def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.333e-06
    I_z = 8.333e-06
    J = 1.666e-05
    n_elements = 10
    element_length = L / n_elements
    axis = np.array([1.0, 1.0, 1.0])
    axis_unit = axis / np.linalg.norm(axis)
    node_coords = []
    for i in range(n_elements + 1):
        node_coords.append(i * element_length * axis_unit)
    node_coords = np.array(node_coords)
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    force_magnitude = 1000.0
    force_direction = np.array([1.0, -1.0, 0.0])
    force_direction = force_direction / np.linalg.norm(force_direction)
    force_vector = force_magnitude * force_direction
    nodal_loads = {n_elements: [force_vector[0], force_vector[1], force_vector[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_node_idx = n_elements
    tip_disp = u[6 * tip_node_idx:6 * tip_node_idx + 3]
    analytical_deflection = force_magnitude * L ** 3 / (3 * E * I_y)
    computed_deflection_magnitude = np.linalg.norm(tip_disp)
    rel_error = abs(computed_deflection_magnitude - analytical_deflection) / analytical_deflection
    assert rel_error < 0.05

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions."""
    node_coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 0.0], [0.0, 5.0, 0.0], [2.5, 2.5, 5.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-06, 'I_z': 8.333e-06, 'J': 1.666e-05, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-06, 'I_z': 8.333e-06, 'J': 1.666e-05, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-06, 'I_z': 8.333e-06, 'J': 1.666e-05, 'local_z': None}, {'node_i': 3, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-06, 'I_z': 8.333e-06, 'J': 1.666e-05, 'local_z': None}, {'node_i': 0, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-06, 'I_z': 8.333e-06, 'J': 1.666e-05, 'local_z': None}, {'node_i': 1, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-06, 'I_z': 8.333e-06, 'J': 1.666e-05, 'local_z': None}, {'node_i': 2, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-06, 'I_z': 8.333e-06, 'J': 1.666e-05, 'local_z': None}, {'node_i': 3, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-06, 'I_z': 8.333e-06, 'J': 1.666e-05, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {4: [10000.0, 5000.0, -20000.0, 1000.0, -2000.0, 3000.0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, {})
    assert np.allclose(u1, 0.0, atol=1e-12)
    assert np.allclose(r1, 0.0, atol=1e-12)
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u2, 0.0, atol=1e-12)
    assert not np.allclose(r2, 0.0, atol=1e-12)
    doubled_loads = {node: [2 * val for val in loads] for (node, loads) in nodal_loads.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, doubled_loads)
    assert np.allclose(u3, 2 * u2, rtol=1e-10)
    assert np.allclose(r3, 2 * r2, rtol=1e-10)
    negated_loads = {node: [-val for val in loads] for (node, loads) in nodal_loads.items()}
    (u4, r4) = fcn(node_coords, elements, boundary_conditions, negated_loads)
    assert np.allclose(u4, -u2, rtol=1e-10)
    assert np.allclose(r4, -r2, rtol=1e-10)
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    for (node_idx, loads) in nodal_loads.items():
        total_force += np.array(loads[:3])
        node_pos = node_coords[node_idx]
        force = np.array(loads[:3])
        moment_load = np.array(loads[3:6])
        total_moment += np.cross(node_pos, force) + moment_load
    support_reactions_force = np.zeros(3)
    support_reactions_moment = np.zeros(3)
    for (node_idx, bc) in boundary_conditions.items():
        if any(bc):
            node_r = r2[6 * node_idx:6 * node_idx + 6]
            support_reactions_force += np.array(node_r[:3])
            support_reactions_moment += np.array(node_r[3:6])
    assert np.allclose(total_force + support_reactions_force, 0.0, atol=1e-10)
    assert np.allclose(total_moment + support_reactions_moment, 0.0, atol=1e-10)