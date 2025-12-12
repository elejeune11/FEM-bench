def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L_total = 3.0
    num_elements = 10
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I = 1e-06
    J = 2e-06
    node_coords_list = []
    for i in range(num_elements + 1):
        node_coords_list.append([i * L_total / num_elements] * 3)
    node_coords = np.array(node_coords_list)
    elements = []
    for i in range(num_elements):
        elem = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J}
        elements.append(elem)
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    perp_vector = np.array([1.0, 1.0, -2.0])
    perp_unit_vector = perp_vector / np.linalg.norm(perp_vector)
    F_magnitude = 1000.0
    force_vector = F_magnitude * perp_unit_vector
    nodal_loads = {10: [force_vector[0], force_vector[1], force_vector[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    L = L_total
    I_eff = I
    delta_analytical_magnitude = F_magnitude * L ** 3 / (3 * E * I_eff)
    u_tip = u[-6:-3]
    delta_computed_magnitude = np.linalg.norm(u_tip)
    assert np.isclose(delta_computed_magnitude, delta_analytical_magnitude, rtol=0.01)

def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-06
    I_z = 1e-06
    J = 2e-06
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 3, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 0, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 1: [1, 1, 1, 1, 1, 1], 2: [1, 1, 1, 1, 1, 1], 3: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0.0)
    assert np.allclose(r_zero, 0.0)
    nodal_loads = {4: [1000.0, 500.0, -2000.0, 100.0, -50.0, 25.0]}
    (u_orig, r_orig) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u_orig, 0.0)
    assert not np.allclose(r_orig, 0.0)
    nodal_loads_double = {4: [2000.0, 1000.0, -4000.0, 200.0, -100.0, 50.0]}
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u_double, 2 * u_orig)
    assert np.allclose(r_double, 2 * r_orig)
    nodal_loads_neg = {4: [-1000.0, -500.0, 2000.0, -100.0, 50.0, -25.0]}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u_neg, -u_orig)
    assert np.allclose(r_neg, -r_orig)
    total_force_applied = np.array([1000.0, 500.0, -2000.0])
    total_moment_applied = np.array([100.0, -50.0, 25.0])
    total_reaction_force = np.zeros(3)
    total_reaction_moment = np.zeros(3)
    n_nodes = node_coords.shape[0]
    for i in range(n_nodes):
        start_idx = i * 6
        total_reaction_force += r_orig[start_idx:start_idx + 3]
        total_reaction_moment += r_orig[start_idx + 3:start_idx + 6]
    assert np.allclose(total_reaction_force, -total_force_applied)
    assert np.allclose(total_reaction_moment, -total_moment_applied)