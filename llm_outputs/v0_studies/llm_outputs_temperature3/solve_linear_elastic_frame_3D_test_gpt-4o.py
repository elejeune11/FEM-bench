def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements,
    tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    node_coords = np.array([[i, i, i] for i in range(11)])
    elements = [{'node_i': i, 'node_j': i + 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None} for i in range(10)]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {10: [0, 0, -1000, 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    L = np.sqrt(3) * 10
    E = 210000000000.0
    I = 1e-06
    F = 1000
    analytical_deflection = F * L ** 3 / (3 * E * I)
    assert np.isclose(u[10 * 6 + 2], analytical_deflection, atol=1e-05), 'Tip deflection does not match analytical solution'

def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    """
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 3, 'node_j': 0, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 0, 'node_j': 4, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u, 0), 'Displacements should be zero with zero loads'
    assert np.allclose(r, 0), 'Reactions should be zero with zero loads'
    nodal_loads = {1: [100, 0, 0, 0, 0, 0], 2: [0, 200, 0, 0, 0, 0], 3: [0, 0, 300, 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u, 0), 'Displacements should be nonzero with applied loads'
    assert not np.allclose(r, 0), 'Reactions should be nonzero with applied loads'
    nodal_loads = {1: [200, 0, 0, 0, 0, 0], 2: [0, 400, 0, 0, 0, 0], 3: [0, 0, 600, 0, 0, 0]}
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u_double, 2 * u), 'Displacements should double with doubled loads'
    assert np.allclose(r_double, 2 * r), 'Reactions should double with doubled loads'
    nodal_loads = {1: [-100, 0, 0, 0, 0, 0], 2: [0, -200, 0, 0, 0, 0], 3: [0, 0, -300, 0, 0, 0]}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u_neg, -u), 'Displacements should flip sign with negated loads'
    assert np.allclose(r_neg, -r), 'Reactions should flip sign with negated loads'
    total_loads = np.zeros(6 * len(node_coords))
    for (node, load) in nodal_loads.items():
        total_loads[node * 6:node * 6 + 6] = load
    assert np.allclose(np.sum(r_neg) + np.sum(total_loads), 0), 'Reactions should satisfy global static equilibrium'