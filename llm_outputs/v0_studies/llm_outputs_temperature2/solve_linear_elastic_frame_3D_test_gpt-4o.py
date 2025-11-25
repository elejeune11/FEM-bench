def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    node_coords = np.array([[i, i, i] for i in range(11)])
    elements = [{'node_i': i, 'node_j': i + 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None} for i in range(10)]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {10: [0, 1000.0, 0, 0, 0, 0]}
    L = np.sqrt(3) * 10
    E = 210000000000.0
    I = 1e-06
    F = 1000.0
    analytical_deflection = F * L ** 3 / (3 * E * I)
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.isclose(u[10 * 6 + 1], analytical_deflection, rtol=0.001), 'Tip deflection does not match analytical solution'

def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    """
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 3, 'node_j': 0, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 4, 'node_j': 5, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 5, 'node_j': 6, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 6, 'node_j': 7, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 7, 'node_j': 4, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 0, 'node_j': 4, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 1, 'node_j': 5, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 2, 'node_j': 6, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 3, 'node_j': 7, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 3: [1, 1, 1, 1, 1, 1], 4: [1, 1, 1, 1, 1, 1], 7: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u, 0), 'Displacements should be zero with zero loads'
    assert np.allclose(r, 0), 'Reactions should be zero with zero loads'
    nodal_loads = {1: [100, 0, 0, 0, 0, 0], 2: [0, 200, 0, 0, 0, 0], 5: [0, 0, 300, 0, 0, 0], 6: [0, 0, 0, 10, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u, 0), 'Displacements should be nonzero with applied loads'
    assert not np.allclose(r, 0), 'Reactions should be nonzero with applied loads'
    u_original = u.copy()
    r_original = r.copy()
    nodal_loads = {1: [200, 0, 0, 0, 0, 0], 2: [0, 400, 0, 0, 0, 0], 5: [0, 0, 600, 0, 0, 0], 6: [0, 0, 0, 20, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u, 2 * u_original), 'Displacements should double with doubled loads'
    assert np.allclose(r, 2 * r_original), 'Reactions should double with doubled loads'
    nodal_loads = {1: [-100, 0, 0, 0, 0, 0], 2: [0, -200, 0, 0, 0, 0], 5: [0, 0, -300, 0, 0, 0], 6: [0, 0, 0, -10, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u, -u_original), 'Displacements should flip sign with negated loads'
    assert np.allclose(r, -r_original), 'Reactions should flip sign with negated loads'
    total_loads = np.zeros(6)
    for load in nodal_loads.values():
        total_loads += np.array(load)
    assert np.allclose(r.sum(axis=0), -total_loads), 'Reactions should satisfy global static equilibrium'