def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = I_z = 1e-05
    J = 2e-05
    node_coords = np.array([[i * L / 10, i * L / 10, i * L / 10] for i in range(11)])
    elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J} for i in range(10)]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F = 1000.0
    nodal_loads = {10: [0, -F / np.sqrt(2), F / np.sqrt(2), 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_deflection = np.sqrt(u[-6] ** 2 + u[-5] ** 2)
    analytical_deflection = F * L ** 3 / (3 * E * I_y)
    assert np.isclose(tip_deflection, analytical_deflection, atol=1e-06)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 3, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 0, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 1: [0, 0, 0, 0, 0, 0], 2: [0, 0, 0, 0, 0, 0], 3: [0, 0, 0, 0, 0, 0], 4: [0, 0, 0, 0, 0, 0]}
    nodal_loads_zero = {i: [0.0] * 6 for i in range(5)}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0)
    assert np.allclose(r_zero, 0)
    nodal_loads = {1: [1000, 0, 0, 0, 0, 0], 2: [0, 1000, 0, 0, 0, 1000], 3: [0, 0, 0, 1000, 0, 0], 4: [0, 0, 1000, 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u, 0)
    assert not np.allclose(r, 0)
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, {k: [2 * x for x in v] for (k, v) in nodal_loads.items()})
    assert np.allclose(u_double, 2 * u)
    assert np.allclose(r_double, 2 * r)
    (u_negate, r_negate) = fcn(node_coords, elements, boundary_conditions, {k: [-x for x in v] for (k, v) in nodal_loads.items()})
    assert np.allclose(u_negate, -u)
    assert np.allclose(r_negate, -r)
    F_global = np.array([sum([v[i] for (k, v) in nodal_loads.items() if k in boundary_conditions and boundary_conditions[k][i] == 0]) for i in range(3)])
    M_global = np.array([sum([v[i + 3] for (k, v) in nodal_loads.items() if k in boundary_conditions and boundary_conditions[k][i + 3] == 0]) for i in range(3)])
    r_sum = np.sum(r.reshape(-1, 6)[:, :3], axis=0)
    m_sum = np.sum(r.reshape(-1, 6)[:, 3:], axis=0)
    assert np.allclose(F_global + r_sum, 0)
    assert np.allclose(M_global + m_sum, 0)