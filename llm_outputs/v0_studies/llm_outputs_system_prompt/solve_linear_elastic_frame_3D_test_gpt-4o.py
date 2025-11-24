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
    nodal_loads = {10: [0, 1000, 0, 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    L = np.sqrt(3) * 10
    F = 1000
    E = 210000000000.0
    I = 1e-06
    analytical_deflection = F * L ** 3 / (3 * E * I)
    assert np.isclose(u[10 * 6 + 1], analytical_deflection, rtol=0.001)

def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    """
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 3, 'node_j': 0, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 0, 'node_j': 4, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u, 0)
    assert np.allclose(r, 0)
    nodal_loads = {1: [100, 0, 0, 0, 0, 0], 2: [0, 200, 0, 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u, 0)
    assert not np.allclose(r, 0)
    nodal_loads = {1: [200, 0, 0, 0, 0, 0], 2: [0, 400, 0, 0, 0, 0]}
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u_double, 2 * u)
    assert np.allclose(r_double, 2 * r)
    nodal_loads = {1: [-100, 0, 0, 0, 0, 0], 2: [0, -200, 0, 0, 0, 0]}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u_neg, -u)
    assert np.allclose(r_neg, -r)
    total_reaction = np.zeros(6)
    for i in range(len(node_coords)):
        total_reaction += r[i * 6:(i + 1) * 6]
    total_load = np.zeros(6)
    for load in nodal_loads.values():
        total_load += load
    assert np.allclose(total_reaction, total_load)