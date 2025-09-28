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
    dir_vec = np.array([1, 1, 1]) / np.linalg.norm([1, 1, 1])
    element_length = L / n_elements
    for i in range(n_nodes):
        node_coords[i] = i * element_length * dir_vec
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    tip_node = n_nodes - 1
    force_dir = np.cross(dir_vec, np.array([1, 0, 0]))
    force_dir = force_dir / np.linalg.norm(force_dir)
    force_magnitude = 1000.0
    nodal_loads = {tip_node: [force_dir[0] * force_magnitude, force_dir[1] * force_magnitude, force_dir[2] * force_magnitude, 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    I_min = min(I_y, I_z)
    analytical_deflection = force_magnitude * L ** 3 / (3 * E * I_min)
    tip_dofs = slice(tip_node * 6, tip_node * 6 + 3)
    numerical_deflection = np.linalg.norm(u[tip_dofs])
    rel_error = abs(numerical_deflection - analytical_deflection) / analytical_deflection
    assert rel_error < 0.02, f'Deflection error too large: {rel_error:.4f}'
    total_force = np.zeros(3)
    for (node, load) in nodal_loads.items():
        total_force += np.array(load[:3])
    total_reaction = np.zeros(3)
    for node in boundary_conditions:
        node_dofs = slice(node * 6, node * 6 + 3)
        total_reaction += r[node_dofs]
    force_balance = np.linalg.norm(total_force + total_reaction)
    assert force_balance < 1e-06, f'Force equilibrium violated: {force_balance}'

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions."""
    node_coords = np.array([[0, 0, 0], [5, 0, 0], [5, 0, 3], [0, 0, 3], [5, 4, 3], [0, 4, 3], [2.5, 2, 1.5]])
    elements = [{'node_i': 0, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 2, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 3, 'node_j': 5, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 4, 'node_j': 5, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 2, 'node_j': 6, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 6, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 1: [1, 1, 1, 1, 1, 1]}
    nodal_loads_stage1 = {}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads_stage1)
    assert np.allclose(u1, 0.0, atol=1e-12), 'Non-zero displacements with zero loads'
    assert np.allclose(r1, 0.0, atol=1e-12), 'Non-zero reactions with zero loads'
    nodal_loads_stage2 = {4: [10.0, -5.0, 3.0, 2.0, -1.0, 0.5], 5: [-8.0, 3.0, -2.0, -1.0, 0.5, -0.2]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_stage2)
    assert not np.allclose(u2, 0.0, atol=1e-12), 'Zero displacements with applied loads'
    assert not np.allclose(r2, 0.0, atol=1e-12), 'Zero reactions with applied loads'
    nodal_loads_stage3 = {4: [20.0, -10.0, 6.0, 4.0, -2.0, 1.0], 5: [-16.0, 6.0, -4.0, -2.0, 1.0, -0.4]}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_stage3)
    assert np.allclose(u3, 2.0 * u2, rtol=1e-10), 'Displacements not proportional to loads'
    assert np.allclose(r3, 2.0 * r2, rtol=1e-10), 'Reactions not proportional to loads'
    nodal_loads_stage4 = {4: [-10.0, 5.0, -3.0, -2.0, 1.0, -0.5], 5: [8.0, -3.0, 2.0, 1.0, -0.5, 0.2]}
    (u4, r4) = fcn(node_coords, elements, boundary_conditions, nodal_loads_stage4)
    assert np.allclose(u4, -u2, rtol=1e-10), 'Displacements not antisymmetric'
    assert np.allclose(r4, -r2, rtol=1e-10), 'Reactions not antisymmetric'
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    for (node, loads) in nodal_loads_stage2.items():
        force = np.array(loads[:3])
        moment = np.array(loads[3:])
        position = node_coords[node]
        total_force += force
        total_moment += moment + np.cross(position, force)
    for node in boundary_conditions:
        reaction_force = r2[node * 6:node * 6 + 3]
        reaction_moment = r2[node * 6 + 3:node * 6 + 6]
        position = node_coords[node]
        total_force += reaction_force
        total_moment += reaction_moment + np.cross(position, reaction_force)
    assert np.linalg.norm(total_force) < 1e-10, 'Force equilibrium not satisfied'
    assert np.linalg.norm(total_moment) < 1e-10, 'Moment equilibrium not satisfied'