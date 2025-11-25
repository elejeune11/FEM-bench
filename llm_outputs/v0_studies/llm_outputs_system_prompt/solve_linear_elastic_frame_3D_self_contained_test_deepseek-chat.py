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
    num_elements = 10
    element_length = L / num_elements
    nodes = []
    for i in range(num_elements + 1):
        t = i / num_elements
        nodes.append([t, t, t])
    node_coords = np.array(nodes)
    elements = []
    for i in range(num_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_tip = 1000.0
    force_direction = np.array([1, -1, 0])
    force_direction = force_direction / np.linalg.norm(force_direction)
    nodal_loads = {num_elements: list(F_tip * force_direction) + [0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_node_idx = num_elements
    tip_disp = u[6 * tip_node_idx:6 * tip_node_idx + 3]
    expected_tip_deflection_magnitude = F_tip * L ** 3 / (3 * E * I_y)
    actual_tip_deflection_magnitude = np.linalg.norm(tip_disp)
    assert np.isclose(actual_tip_deflection_magnitude, expected_tip_deflection_magnitude, rtol=0.05)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0], [0, 5, 0], [2.5, 2.5, 5]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 3, 'node_j': 0, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 0, 'node_j': 4, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 1, 'node_j': 4, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 2, 'node_j': 4, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 3, 'node_j': 4, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0.0)
    assert np.allclose(r_zero, 0.0)
    nodal_loads = {4: [1000.0, 2000.0, 3000.0, 100.0, 200.0, 300.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u, 0.0)
    assert not np.allclose(r, 0.0)
    nodal_loads_double = {4: [2000.0, 4000.0, 6000.0, 200.0, 400.0, 600.0]}
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u_double, 2 * u, rtol=1e-10)
    assert np.allclose(r_double, 2 * r, rtol=1e-10)
    nodal_loads_negated = {4: [-1000.0, -2000.0, -3000.0, -100.0, -200.0, -300.0]}
    (u_negated, r_negated) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negated)
    assert np.allclose(u_negated, -u, rtol=1e-10)
    assert np.allclose(r_negated, -r, rtol=1e-10)
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    for (node_idx, load) in nodal_loads.items():
        force = np.array(load[:3])
        moment = np.array(load[3:])
        position = node_coords[node_idx]
        total_force += force
        total_moment += moment + np.cross(position, force)
    total_reaction_force = np.zeros(3)
    total_reaction_moment = np.zeros(3)
    for (node_idx, bc) in boundary_conditions.items():
        if any(bc[:3]):
            reaction_start = 6 * node_idx
            reaction_force = r[reaction_start:reaction_start + 3]
            reaction_moment = r[reaction_start + 3:reaction_start + 6]
            position = node_coords[node_idx]
            total_reaction_force += reaction_force
            total_reaction_moment += reaction_moment + np.cross(position, reaction_force)
    assert np.allclose(total_force + total_reaction_force, 0.0, atol=1e-10)
    assert np.allclose(total_moment + total_reaction_moment, 0.0, atol=1e-10)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff).
    The solver should detect this (cond(K_ff) >= 1e16) and raise a ValueError."""
    node_coords = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}]
    boundary_conditions = {}
    nodal_loads = {2: [1000.0, 0, 0, 0, 0, 0]}
    with pytest.raises(ValueError):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)