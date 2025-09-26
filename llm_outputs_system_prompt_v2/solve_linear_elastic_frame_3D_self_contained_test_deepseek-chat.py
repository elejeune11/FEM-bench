def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
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
    element_length = L / n_elements
    beam_axis = np.array([1.0, 1.0, 1.0])
    beam_axis /= np.linalg.norm(beam_axis)
    nodes = []
    for i in range(n_elements + 1):
        nodes.append(i * element_length * beam_axis)
    node_coords = np.array(nodes)
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    force_direction = np.cross(beam_axis, np.array([0.0, 0.0, 1.0]))
    force_direction /= np.linalg.norm(force_direction)
    F_mag = 1000.0
    force = F_mag * force_direction
    nodal_loads = {n_elements: [force[0], force[1], force[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_node = n_elements
    tip_disp = u[6 * tip_node:6 * tip_node + 3]
    analytical_deflection = F_mag * L ** 3 / (3 * E * I_y)
    computed_deflection_mag = np.linalg.norm(tip_disp)
    assert np.isclose(computed_deflection_mag, analytical_deflection, rtol=0.05)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 0.0], [0.0, 5.0, 0.0], [2.5, 2.5, 5.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 3, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 0, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 1, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 2, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 3, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 1: [1, 1, 1, 1, 1, 1], 2: [1, 1, 1, 1, 1, 1], 3: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0.0)
    assert np.allclose(r_zero, 0.0)
    nodal_loads = {4: [10000.0, 5000.0, -2000.0, 1000.0, -500.0, 200.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u, 0.0)
    assert not np.allclose(r, 0.0)
    nodal_loads_double = {4: [20000.0, 10000.0, -4000.0, 2000.0, -1000.0, 400.0]}
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u_double, 2.0 * u, rtol=1e-10)
    assert np.allclose(r_double, 2.0 * r, rtol=1e-10)
    nodal_loads_neg = {4: [-10000.0, -5000.0, 2000.0, -1000.0, 500.0, -200.0]}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u_neg, -u, rtol=1e-10)
    assert np.allclose(r_neg, -r, rtol=1e-10)
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    for (node_idx, load) in nodal_loads.items():
        force = np.array(load[:3])
        moment = np.array(load[3:])
        total_force += force
        total_moment += moment + np.cross(node_coords[node_idx], force)
    reaction_forces = np.zeros(3)
    reaction_moments = np.zeros(3)
    for node_idx in boundary_conditions:
        if node_idx in boundary_conditions:
            reaction_forces += r[6 * node_idx:6 * node_idx + 3]
            reaction_moments += r[6 * node_idx + 3:6 * node_idx + 6] + np.cross(node_coords[node_idx], r[6 * node_idx:6 * node_idx + 3])
    assert np.allclose(total_force + reaction_forces, 0.0, atol=1e-10)
    assert np.allclose(total_moment + reaction_moments, 0.0, atol=1e-10)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff).
    The solver should detect this (cond(K_ff) >= 1e16) and raise a ValueError."""
    node_coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}]
    boundary_conditions = {}
    nodal_loads = {2: [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    with pytest.raises(ValueError):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)