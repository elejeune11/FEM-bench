def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L_total = 10.0
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-06
    I_z = 1e-06
    J = 2e-06
    n_elements = 10
    n_nodes = n_elements + 1
    node_coords = np.zeros((n_nodes, 3))
    direction = np.array([1, 1, 1]) / np.sqrt(3)
    L_element = L_total / n_elements
    for i in range(n_nodes):
        node_coords[i] = i * L_element * direction
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0, 0, 1])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_magnitude = 1000.0
    force_dir = np.cross(direction, np.array([1, 0, 0]))
    force_dir = force_dir / np.linalg.norm(force_dir)
    force_vector = F_magnitude * force_dir
    nodal_loads = {n_nodes - 1: [force_vector[0], force_vector[1], force_vector[2], 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    I_effective = max(I_y, I_z)
    analytical_deflection = F_magnitude * L_total ** 3 / (3 * E * I_effective)
    tip_disp = u[(n_nodes - 1) * 6:(n_nodes - 1) * 6 + 3]
    computed_deflection = np.linalg.norm(tip_disp)
    rel_error = abs(computed_deflection - analytical_deflection) / analytical_deflection
    assert rel_error < 0.01, f'Deflection error too large: {rel_error:.3f}'
    total_applied_force = np.array([force_vector[0], force_vector[1], force_vector[2]])
    total_reaction_force = np.zeros(3)
    for node in range(n_nodes):
        if node in boundary_conditions and any(boundary_conditions[node]):
            reaction_start = node * 6
            total_reaction_force += r[reaction_start:reaction_start + 3]
    force_balance_error = np.linalg.norm(total_applied_force + total_reaction_force)
    assert force_balance_error < 1e-06, 'Force equilibrium not satisfied'

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions."""
    node_coords = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5], [5, 5, 5]])
    elements = []
    for i in range(4):
        elements.append({'node_i': i, 'node_j': (i + 1) % 5 if i == 3 else i + 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.02, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0, 0, 1])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u1, 0), 'Non-zero displacements with zero loads'
    assert np.allclose(r1, 0), 'Non-zero reactions with zero loads'
    nodal_loads = {1: [1000, 500, -200, 0, 0, 0], 2: [0, 0, 0, 500, -300, 200], 4: [-500, 1000, 300, 0, 100, 0]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u2, 0), 'Zero displacements with applied loads'
    assert not np.allclose(r2, 0), 'Zero reactions with applied loads'
    nodal_loads_double = {node: [2 * val for val in load] for (node, load) in nodal_loads.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u3, 2 * u2, rtol=1e-10), 'Linearity violation in displacements'
    assert np.allclose(r3, 2 * r2, rtol=1e-10), 'Linearity violation in reactions'
    nodal_loads_negated = {node: [-val for val in load] for (node, load) in nodal_loads.items()}
    (u4, r4) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negated)
    assert np.allclose(u4, -u2, rtol=1e-10), 'Sign flip violation in displacements'
    assert np.allclose(r4, -r2, rtol=1e-10), 'Sign flip violation in reactions'
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    for (node, load) in nodal_loads.items():
        force = np.array(load[:3])
        moment = np.array(load[3:])
        position = node_coords[node]
        total_force += force
        total_moment += moment + np.cross(position, force)
    total_reaction_force = np.zeros(3)
    total_reaction_moment = np.zeros(3)
    for node in range(len(node_coords)):
        if node in boundary_conditions and any(boundary_conditions[node]):
            reaction_start = node * 6
            reaction_force = r2[reaction_start:reaction_start + 3]
            reaction_moment = r2[reaction_start + 3:reaction_start + 6]
            position = node_coords[node]
            total_reaction_force += reaction_force
            total_reaction_moment += reaction_moment + np.cross(position, reaction_force)
    assert np.allclose(total_force + total_reaction_force, 0, atol=1e-06), 'Force equilibrium not satisfied'
    assert np.allclose(total_moment + total_reaction_moment, 0, atol=1e-06), 'Moment equilibrium not satisfied'

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff)."""
    node_coords = np.array([[0, 0, 0], [2, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06, 'local_z': np.array([0, 0, 1])}]
    boundary_conditions = {}
    nodal_loads = {1: [1000, 0, 0, 0, 0, 0]}
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)