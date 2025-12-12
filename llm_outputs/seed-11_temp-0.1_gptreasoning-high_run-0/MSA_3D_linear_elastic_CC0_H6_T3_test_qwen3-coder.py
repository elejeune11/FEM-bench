def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-05
    I_z = 1e-05
    J = 1e-05
    num_elements = 10
    total_length = 3.0
    node_coords = np.array([[i * total_length / num_elements, i * total_length / num_elements, i * total_length / num_elements] for i in range(num_elements + 1)])
    elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J} for i in range(num_elements)]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    tip_node = len(node_coords) - 1
    F_transverse = 1000.0
    nodal_loads = {tip_node: [0.0, 0.0, F_transverse, 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    L = total_length
    delta_analytical = F_transverse * L ** 3 / (3 * E * I_y_eq)
    I_y_eq = 1e-05
    delta_analytical = F_transverse * L ** 3 / (3 * E * I_y_eq)
    idx_tip_disp_z = tip_node * 6 + 2
    computed_deflection = u[idx_tip_disp_z]
    assert abs(computed_deflection - delta_analytical) / abs(delta_analytical) < 0.05

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
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 1.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'node_i': 3, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'node_i': 0, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'node_i': 1, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'node_i': 2, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'node_i': 3, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 1: [1, 1, 1, 1, 1, 1], 2: [1, 1, 1, 1, 1, 1], 3: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0.0), 'Displacements should be zero with zero loads'
    assert np.allclose(r_zero, 0.0), 'Reactions should be zero with zero loads'
    nodal_loads_original = {4: [1000.0, 500.0, -2000.0, 100.0, -50.0, 25.0]}
    (u_orig, r_orig) = fcn(node_coords, elements, boundary_conditions, nodal_loads_original)
    assert not np.allclose(u_orig, 0.0), 'Displacements should be nonzero with applied loads'
    assert not np.allclose(r_orig, 0.0), 'Reactions should be nonzero with applied loads'
    nodal_loads_doubled = {4: [2000.0, 1000.0, -4000.0, 200.0, -100.0, 50.0]}
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, nodal_loads_doubled)
    assert np.allclose(u_double, 2 * u_orig), 'Displacements should double when loads are doubled'
    assert np.allclose(r_double, 2 * r_orig), 'Reactions should double when loads are doubled'
    nodal_loads_negated = {4: [-1000.0, -500.0, 2000.0, -100.0, 50.0, -25.0]}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negated)
    assert np.allclose(u_neg, -u_orig), 'Displacements should negate when loads are negated'
    assert np.allclose(r_neg, -r_orig), 'Reactions should negate when loads are negated'
    total_applied_force = np.zeros(6)
    for (node_idx, load) in nodal_loads_original.items():
        total_applied_force += np.array(load)
    total_reaction_force = np.zeros(6)
    for node_idx in boundary_conditions:
        start_idx = node_idx * 6
        end_idx = start_idx + 6
        total_reaction_force += r_orig[start_idx:end_idx]
    assert np.allclose(total_reaction_force, -total_applied_force), 'Reactions should balance applied loads for global equilibrium'