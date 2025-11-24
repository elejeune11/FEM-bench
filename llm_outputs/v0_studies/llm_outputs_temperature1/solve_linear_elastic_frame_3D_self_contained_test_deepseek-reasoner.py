def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.33e-06
    I_z = 8.33e-06
    J = 1.67e-05
    n_elements = 10
    n_nodes = n_elements + 1
    node_coords = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        factor = i * L / n_elements / np.sqrt(3)
        node_coords[i] = [factor, factor, factor]
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    tip_force_magnitude = 1000.0
    force_dir = np.array([1, -1, 0])
    force_dir = force_dir / np.linalg.norm(force_dir) * tip_force_magnitude
    nodal_loads = {n_nodes - 1: [force_dir[0], force_dir[1], force_dir[2], 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    actual_length = L
    I_effective = (I_y + I_z) / 2
    expected_deflection = tip_force_magnitude * actual_length ** 3 / (3 * E * I_effective)
    tip_node_idx = n_nodes - 1
    tip_disp = u[tip_node_idx * 6:tip_node_idx * 6 + 3]
    actual_deflection = np.linalg.norm(tip_disp)
    assert abs(actual_deflection - expected_deflection) / expected_deflection < 0.01

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions."""
    node_coords = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0], [5, 5, 0], [0, 0, 5], [5, 0, 5]])
    elements = []
    element_props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}
    connections = [(0, 1), (0, 2), (1, 3), (2, 3), (0, 4), (1, 5), (4, 5), (2, 4), (3, 5)]
    for (i, j) in connections:
        elements.append({'node_i': i, 'node_j': j, **element_props})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 1: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0, atol=1e-10)
    assert np.linalg.norm(r_zero) < 1e-10
    nodal_loads = {2: [1000, 2000, -1500, 500, -300, 400], 3: [-500, 1000, 800, -200, 150, -100], 4: [0, 0, -2000, 0, 0, 0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u1, 0, atol=1e-10)
    assert not np.allclose(r1, 0, atol=1e-10)
    nodal_loads_double = {node: [2 * x for x in loads] for (node, loads) in nodal_loads.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(2 * u1, u2, rtol=1e-10)
    assert np.allclose(2 * r1, r2, rtol=1e-10)
    nodal_loads_negate = {node: [-x for x in loads] for (node, loads) in nodal_loads.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negate)
    assert np.allclose(-u1, u3, rtol=1e-10)
    assert np.allclose(-r1, r3, rtol=1e-10)
    total_force_applied = np.zeros(3)
    total_moment_applied = np.zeros(3)
    for (node, loads) in nodal_loads.items():
        force = loads[:3]
        moment = loads[3:]
        total_force_applied += force
        total_moment_applied += moment
        pos = node_coords[node]
        total_moment_applied += np.cross(pos, force)
    total_force_reaction = np.zeros(3)
    total_moment_reaction = np.zeros(3)
    for node in range(len(node_coords)):
        start_idx = node * 6
        reaction_force = r1[start_idx:start_idx + 3]
        reaction_moment = r1[start_idx + 3:start_idx + 6]
        total_force_reaction += reaction_force
        total_moment_reaction += reaction_moment
        pos = node_coords[node]
        total_moment_reaction += np.cross(pos, reaction_force)
    assert np.allclose(total_force_applied + total_force_reaction, 0, atol=1e-10)
    assert np.allclose(total_moment_applied + total_moment_reaction, 0, atol=1e-10)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff)."""
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}]
    boundary_conditions = {}
    nodal_loads = {0: [1000, 0, 0, 0, 0, 0], 1: [0, 1000, 0, 0, 0, 0]}
    with pytest.raises(ValueError, match='ill-conditioned|condition number'):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)