def test_simple_beam_discretized_axis_111(fcn):
    node_coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10]])
    elements = [{'node_i': i, 'node_j': i + 1, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1} for i in range(10)]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {10: [0, 10, 0, 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u[18:21], [0, 0, 0], atol=0.1)

def test_complex_geometry_and_basic_loading(fcn):
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}, {'node_i': 1, 'node_j': 2, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}, {'node_i': 2, 'node_j': 3, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}, {'node_i': 3, 'node_j': 0, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}, {'node_i': 0, 'node_j': 4, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {1: [1, 0, 0, 0, 0, 0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    nodal_loads = {1: [2, 0, 0, 0, 0, 0]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    nodal_loads = {1: [-1, 0, 0, 0, 0, 0]}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u1, -u3)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    assert np.allclose(np.sum(r1), 0, atol=1e-06)