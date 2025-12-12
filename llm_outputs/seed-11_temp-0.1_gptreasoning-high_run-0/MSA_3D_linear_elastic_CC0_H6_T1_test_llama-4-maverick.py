def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    node_coords = np.array([[0, 0, 0], [1 / 3, 1 / 3, 1 / 3], [2 / 3, 2 / 3, 2 / 3], [1, 1, 1], [4 / 3, 4 / 3, 4 / 3], [5 / 3, 5 / 3, 5 / 3], [2, 2, 2], [7 / 3, 7 / 3, 7 / 3], [8 / 3, 8 / 3, 8 / 3], [3, 3, 3], [10 / 3, 10 / 3, 10 / 3]])
    elements = [{'node_i': i, 'node_j': i + 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05} for i in range(10)]
    for elem in elements:
        elem['local_z'] = [0, 0, 1]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {10: [0, 100, 0, 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_deflection = np.linalg.norm(u[-6:-3])
    L = np.linalg.norm(node_coords[-1] - node_coords[0])
    E = elements[0]['E']
    I = elements[0]['I_y']
    F = 100
    analytical_deflection = F * L ** 3 / (3 * E * I)
    assert np.isclose(tip_deflection, analytical_deflection, atol=1e-06)

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
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 3, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 4, 'node_j': 5, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 5, 'node_j': 6, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 6, 'node_j': 7, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 7, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 0, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 1, 'node_j': 5, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 2, 'node_j': 6, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 3, 'node_j': 7, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}]
    for elem in elements:
        elem['local_z'] = [0, 0, 1]
    boundary_conditions = {i: [1, 1, 1, 1, 1, 1] for i in range(4)}
    nodal_loads_zero = {i: [0, 0, 0, 0, 0, 0] for i in range(4, 8)}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0)
    assert np.allclose(r_zero, 0)
    nodal_loads = {4: [100, 0, 0, 0, 0, 0], 5: [0, 100, 0, 0, 0, 0], 6: [0, 0, 100, 0, 0, 0], 7: [0, 0, 0, 100, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u, 0)
    assert not np.allclose(r, 0)
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, {k: [2 * x for x in v] for (k, v) in nodal_loads.items()})
    assert np.allclose(u_double, 2 * u)
    assert np.allclose(r_double, 2 * r)
    (u_negate, r_negate) = fcn(node_coords, elements, boundary_conditions, {k: [-x for x in v] for (k, v) in nodal_loads.items()})
    assert np.allclose(u_negate, -u)
    assert np.allclose(r_negate, -r)
    global_forces = np.sum([np.array(nodal_loads.get(i, [0] * 6)) for i in range(8)], axis=0)
    reaction_forces = np.sum([r[i * 6:(i + 1) * 6] for i in range(4)], axis=0)
    assert np.allclose(global_forces[:3] + reaction_forces[:3], 0, atol=1e-06)
    assert np.allclose(global_forces[3:] + reaction_forces[3:], 0, atol=1e-06)