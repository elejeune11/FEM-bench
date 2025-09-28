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
    assert np.isclose(u[10 * 6 + 1], analytical_deflection, atol=0.001), 'Tip deflection does not match analytical solution'

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
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}, {'node_i': 3, 'node_j': 0, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u, 0), 'Displacements are not zero with zero loads'
    assert np.allclose(r, 0), 'Reactions are not zero with zero loads'
    nodal_loads = {1: [100, 200, 300, 10, 20, 30], 2: [-100, -200, -300, -10, -20, -30]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u, 0), 'Displacements are zero with non-zero loads'
    assert not np.allclose(r, 0), 'Reactions are zero with non-zero loads'
    nodal_loads_doubled = {1: [200, 400, 600, 20, 40, 60], 2: [-200, -400, -600, -20, -40, -60]}
    (u_doubled, r_doubled) = fcn(node_coords, elements, boundary_conditions, nodal_loads_doubled)
    assert np.allclose(u_doubled, 2 * u), 'Displacements do not double with doubled loads'
    assert np.allclose(r_doubled, 2 * r), 'Reactions do not double with doubled loads'
    nodal_loads_negated = {1: [-100, -200, -300, -10, -20, -30], 2: [100, 200, 300, 10, 20, 30]}
    (u_negated, r_negated) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negated)
    assert np.allclose(u_negated, -u), 'Displacements do not flip sign with negated loads'
    assert np.allclose(r_negated, -r), 'Reactions do not flip sign with negated loads'
    total_loads = np.zeros(6)
    for load in nodal_loads.values():
        total_loads += np.array(load)
    assert np.allclose(np.sum(r, axis=0), total_loads), 'Reactions do not satisfy global static equilibrium'