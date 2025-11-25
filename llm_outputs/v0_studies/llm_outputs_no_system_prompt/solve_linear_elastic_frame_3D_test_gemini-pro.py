def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L = 10.0
    E = 1.0
    nu = 0.3
    A = 1.0
    Iy = 1.0
    Iz = 1.0
    J = 1.0
    P = 1.0
    n_elements = 10
    node_coords = np.array([[i * L / n_elements, i * L / n_elements, i * L / n_elements] for i in range(n_elements + 1)])
    elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': None} for i in range(n_elements)]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {n_elements: [0, P, 0, 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    I = (Iy + Iz) / 2
    analytical_deflection = P * L ** 3 / (3 * E * I) / np.sqrt(2)
    tip_deflection = u[n_elements * 6 + 1]
    assert np.isclose(tip_deflection, analytical_deflection, rtol=0.01)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium
    """
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 1, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 1, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': 1, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1, 'local_z': None}]
    boundary_conditions = {0: [1] * 6}
    n_nodes = len(node_coords)
    nodal_loads = {}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u, 0)
    assert np.allclose(r, 0)
    nodal_loads = {1: [1, 2, 3, 4, 5, 6], 2: [7, 8, 9, 10, 11, 12]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u1, 0)
    assert not np.allclose(r1, 0)
    nodal_loads_doubled = {node: [2 * val for val in loads] for (node, loads) in nodal_loads.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_doubled)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    nodal_loads_negated = {node: [-val for val in loads] for (node, loads) in nodal_loads.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negated)
    assert np.allclose(u3, -u1)
    assert np.allclose(r3, -r1)
    assert np.allclose(r1.reshape(n_nodes, 6).sum(axis=0)[:3], -np.array([1 + 7, 2 + 8, 3 + 9]))
    assert np.allclose(r1.reshape(n_nodes, 6).sum(axis=0)[3:], -np.array([4 + 10, 5 + 11, 6 + 12]))