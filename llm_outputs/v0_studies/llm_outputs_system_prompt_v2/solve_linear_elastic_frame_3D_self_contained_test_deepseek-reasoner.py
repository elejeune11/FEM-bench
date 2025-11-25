def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    E = 200000000000.0
    I = 0.0001
    P = 1000.0
    n_nodes = 11
    node_coords = np.array([(i * L / 10, i * L / 10, i * L / 10) for i in range(n_nodes)])
    elements = []
    for i in range(n_nodes - 1):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': 0.3, 'A': 0.01, 'I_y': I, 'I_z': I, 'J': 0.0002, 'local_z': np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    force_dir = np.array([1.0, -1.0, 0.0])
    force_dir /= np.linalg.norm(force_dir)
    force_magnitude = P
    force_vector = force_dir * force_magnitude
    nodal_loads = {n_nodes - 1: [force_vector[0], force_vector[1], force_vector[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    beam_length = np.sqrt(3) * L
    analytical_deflection = P * beam_length ** 3 / (3 * E * I)
    tip_node_index = n_nodes - 1
    tip_displacement = u[6 * tip_node_index:6 * tip_node_index + 3]
    computed_deflection = np.dot(tip_displacement, force_dir)
    assert np.abs(computed_deflection - analytical_deflection) / analytical_deflection < 0.01

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions."""
    node_coords = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0], [0, 5, 0], [0, 0, 5], [5, 0, 5], [5, 5, 5], [0, 5, 5]])
    elements = []
    beam_props = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    connections = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for (i, j) in connections:
        elements.append({'node_i': i, 'node_j': j, 'local_z': None, **beam_props})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 3: [1, 1, 1, 0, 0, 0]}
    stage1_loads = {}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, stage1_loads)
    assert np.allclose(u1, 0.0)
    assert np.allclose(r1, 0.0)
    stage2_loads = {4: [1000, 2000, 3000, 400, 500, 600], 5: [-500, 1000, -1500, 200, -300, 400], 6: [0, 0, -5000, 100, 200, -300]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, stage2_loads)
    assert not np.allclose(u2, 0.0)
    assert not np.allclose(r2, 0.0)
    stage3_loads = {node: [2 * x for x in loads] for (node, loads) in stage2_loads.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, stage3_loads)
    assert np.allclose(2 * u2, u3, rtol=1e-10)
    assert np.allclose(2 * r2, r3, rtol=1e-10)
    stage4_loads = {node: [-x for x in loads] for (node, loads) in stage2_loads.items()}
    (u4, r4) = fcn(node_coords, elements, boundary_conditions, stage4_loads)
    assert np.allclose(-u2, u4, rtol=1e-10)
    assert np.allclose(-r2, r4, rtol=1e-10)
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    for (node, loads) in stage2_loads.items():
        force = np.array(loads[:3])
        moment = np.array(loads[3:])
        total_force += force
        total_moment += moment + np.cross(node_coords[node], force)
    fixed_nodes = [0, 3]
    for node in fixed_nodes:
        start_idx = 6 * node
        reaction_force = np.array(r2[start_idx:start_idx + 3])
        reaction_moment = np.array(r2[start_idx + 3:start_idx + 6])
        total_force += reaction_force
        total_moment += reaction_moment + np.cross(node_coords[node], reaction_force)
    assert np.allclose(total_force, 0.0, atol=1e-10)
    assert np.allclose(total_moment, 0.0, atol=1e-10)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff)."""
    node_coords = np.array([[0, 0, 0], [5, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': None}]
    boundary_conditions = {}
    nodal_loads = {1: [1000, 0, 0, 0, 0, 0]}
    with pytest.raises(ValueError):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)