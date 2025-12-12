def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    node_coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.001, 'I_z': 0.001, 'J': 0.002}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.001, 'I_z': 0.001, 'J': 0.002}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.001, 'I_z': 0.001, 'J': 0.002}, {'node_i': 3, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.001, 'I_z': 0.001, 'J': 0.002}, {'node_i': 4, 'node_j': 5, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.001, 'I_z': 0.001, 'J': 0.002}, {'node_i': 5, 'node_j': 6, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.001, 'I_z': 0.001, 'J': 0.002}, {'node_i': 6, 'node_j': 7, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.001, 'I_z': 0.001, 'J': 0.002}, {'node_i': 7, 'node_j': 8, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.001, 'I_z': 0.001, 'J': 0.002}, {'node_i': 8, 'node_j': 9, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.001, 'I_z': 0.001, 'J': 0.002}, {'node_i': 9, 'node_j': 10, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.001, 'I_z': 0.001, 'J': 0.002}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {10: [0, 1000, 0, 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_displacement = u[6 * 10:6 * 10 + 3]
    analytical_displacement = np.array([0.002083, 0.002083, 0.002083])
    assert np.allclose(tip_displacement, analytical_displacement, atol=0.001)

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
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.001, 'I_z': 0.001, 'J': 0.002}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.001, 'I_z': 0.001, 'J': 0.002}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.001, 'I_z': 0.001, 'J': 0.002}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u, 0)
    assert np.allclose(r, 0)
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 3: [0, 0, 0, 0, 0, 0]}
    nodal_loads = {2: [1000, 0, 0, 0, 0, 0], 3: [0, 1000, 0, 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u, 0)
    assert not np.allclose(r, 0)
    nodal_loads = {2: [2000, 0, 0, 0, 0, 0], 3: [0, 2000, 0, 0, 0, 0]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u2, 2 * u)
    assert np.allclose(r2, 2 * r)
    nodal_loads = {2: [-1000, 0, 0, 0, 0, 0], 3: [0, -1000, 0, 0, 0, 0]}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u3, -u)
    assert np.allclose(r3, -r)
    reaction_forces = r
    load_forces = np.zeros(12)
    for (node, loads) in nodal_loads.items():
        load_forces[6 * node:6 * node + 6] = loads
    assert np.allclose(np.sum(reaction_forces + load_forces), 0, atol=1e-06)