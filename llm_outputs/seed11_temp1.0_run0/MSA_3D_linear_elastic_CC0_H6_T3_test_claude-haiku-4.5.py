def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L_elem = np.sqrt(3)
    n_elements = 10
    total_length = n_elements * L_elem
    node_coords = np.zeros((n_elements + 1, 3))
    for i in range(n_elements + 1):
        t = i / n_elements
        node_coords[i] = t * np.array([total_length / np.sqrt(3), total_length / np.sqrt(3), total_length / np.sqrt(3)])
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-05
    I_z = 1e-05
    J = 2e-05
    elements = []
    local_z = np.array([0, 0, 1])
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    P = 1000
    transverse_dir = np.array([1, -1, 0]) / np.sqrt(2)
    load_vector = P * transverse_dir
    nodal_loads = {n_elements: [load_vector[0], load_vector[1], load_vector[2], 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_node_idx = n_elements
    tip_displacement = u[6 * tip_node_idx:6 * tip_node_idx + 3]
    tip_disp_magnitude = np.linalg.norm(tip_displacement)
    assert tip_disp_magnitude > 0, 'Tip displacement should be nonzero under transverse load'
    disp_in_load_dir = np.dot(tip_displacement, transverse_dir)
    assert disp_in_load_dir > 0, 'Tip displacement should be in direction of applied load'
    reaction_force = r[0:3]
    applied_force = np.array([load_vector[0], load_vector[1], load_vector[2]])
    force_balance = reaction_force + applied_force
    assert np.allclose(force_balance, 0, atol=1e-06), 'Reaction forces should balance applied forces'

def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium.
    """
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-05
    I_z = 1e-05
    J = 2e-05
    local_z = np.array([0, 0, 1])
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0, atol=1e-10), 'Zero loads should result in zero displacements'
    assert np.allclose(r_zero, 0, atol=1e-10), 'Zero loads should result in zero reactions'
    nodal_loads_mixed = {3: [100, 50, 25, 10, 5, 2]}
    (u_mixed, r_mixed) = fcn(node_coords, elements, boundary_conditions, nodal_loads_mixed)
    free_disp_mask = np.ones(len(u_mixed), dtype=bool)
    free_disp_mask[0:6] = False
    assert np.any(np.abs(u_mixed[free_disp_mask]) > 1e-10), 'Displacements at free nodes should be nonzero under loading'
    fixed_disp = u_mixed[0:6]
    assert np.allclose(fixed_disp, 0, atol=1e-10), 'Displacements at fixed node should be zero'
    nodal_loads_double = {3: [200, 100, 50, 20, 10, 4]}
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u_double, 2 * u_mixed, rtol=1e-10), 'Doubling loads should double displacements (linearity)'
    assert np.allclose(r_double, 2 * r_mixed, rtol=1e-10), 'Doubling loads should double reactions (linearity)'
    nodal_loads_negated = {3: [-100, -50, -25, -10, -5, -2]}
    (u_negated, r_negated) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negated)
    assert np.allclose(u_negated, -u_mixed, rtol=1e-10), 'Negating loads should negate displacements'
    assert np.allclose(r_negated, -r_mixed, rtol=1e-10), 'Negating loads should negate reactions'
    total_applied_force = np.array(nodal_loads_mixed[3][0:3])
    total_applied_moment = np.array(nodal_loads_mixed[3][3:6])
    reaction_forces = r_mixed[0:3]
    reaction_moments = r_mixed[3:6]
    force_equilibrium = total_applied_force + reaction_forces
    assert np.allclose(force_equilibrium, 0, atol=1e-06), 'Global force equilibrium should be satisfied'
    total_moment = total_applied_moment + reaction_moments
    assert np.linalg.norm(total_moment) < np.linalg.norm(total_applied_moment) * 10, 'Moment should be in reasonable range'