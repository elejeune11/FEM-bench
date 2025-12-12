def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L_total = 10.0
    E = 1.0
    nu = 0.3
    A = 1.0
    I_y = 1.0
    I_z = 1.0
    J = 1.0
    n_elements = 10
    n_nodes = n_elements + 1
    node_coords = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        t = i / n_elements
        node_coords[i] = t * L_total / np.sqrt(3) * np.array([1, 1, 1])
    elements = []
    local_z = np.array([0, 0, 1]) / np.sqrt(1)
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    load_direction = np.array([1, -1, 0]) / np.sqrt(2)
    load_magnitude = 1.0
    nodal_loads = {n_nodes - 1: list(load_magnitude * load_direction) + [0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_node_idx = n_nodes - 1
    tip_displacement = u[6 * tip_node_idx:6 * tip_node_idx + 3]
    expected_deflection_magnitude = load_magnitude * L_total ** 3 / (3 * E * I_y)
    actual_displacement_magnitude = np.linalg.norm(tip_displacement)
    assert np.isclose(actual_displacement_magnitude, expected_deflection_magnitude, rtol=0.05), f'Tip deflection {actual_displacement_magnitude} does not match expected {expected_deflection_magnitude}'
    fixed_node_displacement = u[0:6]
    assert np.allclose(fixed_node_displacement, 0.0), 'Fixed node should have zero displacement'

def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions satisfy global static equilibrium.
    """
    node_coords = np.array([[0, 0, 0], [0, 0, 5], [5, 0, 5]], dtype=float)
    E = 100000.0
    nu = 0.3
    A = 10.0
    I_y = 100.0
    I_z = 100.0
    J = 50.0
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([1, 0, 0])}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0, 0, 1])}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u_zero, 0.0, atol=1e-10), 'Zero loads should result in zero displacements and reactions'
    assert np.allclose(r_zero, 0.0, atol=1e-10), 'Zero loads should result in zero reactions'
    nodal_loads_1 = {2: [100.0, 50.0, -30.0, 10.0, 20.0, 5.0]}
    (u_1, r_1) = fcn(node_coords, elements, boundary_conditions, nodal_loads_1)
    assert not np.allclose(u_1[6 * 2:6 * 3], 0.0, atol=1e-06), 'Node 2 should have nonzero displacements under applied load'
    assert not np.allclose(r_1[0:6], 0.0, atol=1e-06), 'Fixed support should have nonzero reactions'
    nodal_loads_2 = {2: [200.0, 100.0, -60.0, 20.0, 40.0, 10.0]}
    (u_2, r_2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_2)
    assert np.allclose(u_2, 2 * u_1, rtol=1e-10), 'Doubling loads should double displacements (linearity)'
    assert np.allclose(r_2, 2 * r_1, rtol=1e-10), 'Doubling loads should double reactions (linearity)'
    nodal_loads_neg = {2: [-100.0, -50.0, 30.0, -10.0, -20.0, -5.0]}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u_neg, -u_1, rtol=1e-10), 'Negating loads should negate displacements'
    assert np.allclose(r_neg, -r_1, rtol=1e-10), 'Negating loads should negate reactions'
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    for (node_idx, loads) in nodal_loads_1.items():
        total_force += np.array(loads[0:3])
        total_moment += np.array(loads[3:6])
    for (node_idx, bc) in boundary_conditions.items():
        if all(bc):
            total_force += r_1[6 * node_idx:6 * node_idx + 3]
            total_moment += r_1[6 * node_idx + 3:6 * node_idx + 6]
    assert np.allclose(total_force, 0.0, atol=1e-06), 'Global force equilibrium not satisfied'
    assert np.allclose(total_moment, 0.0, atol=1e-06), 'Global moment equilibrium not satisfied'