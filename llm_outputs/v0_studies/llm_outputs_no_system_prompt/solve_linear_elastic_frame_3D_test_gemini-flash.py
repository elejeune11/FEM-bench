def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    E = 200000000000.0
    A = 0.01
    I = 1e-05
    F = 1000.0
    node_coords = np.array([[0, 0, 0], [L / 10, L / 10, L / 10], [2 * L / 10, 2 * L / 10, 2 * L / 10], [3 * L / 10, 3 * L / 10, 3 * L / 10], [4 * L / 10, 4 * L / 10, 4 * L / 10], [5 * L / 10, 5 * L / 10, 5 * L / 10], [6 * L / 10, 6 * L / 10, 6 * L / 10], [7 * L / 10, 7 * L / 10, 7 * L / 10], [8 * L / 10, 8 * L / 10, 8 * L / 10], [9 * L / 10, 9 * L / 10, 9 * L / 10], [L, L, L]])
    elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': 0.3, 'A': A, 'I_y': I, 'I_z': I, 'J': I} for i in range(10)]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {10: [F, 0, 0, 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    analytical_tip_deflection = F * L ** 3 / (3 * E * I) * 1 / np.sqrt(3)
    assert u[30] == approx(analytical_tip_deflection, rel=0.1)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'node_i': 3, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'node_i': 0, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_1 = {1: [100, 0, 0, 0, 0, 0], 2: [0, 100, 0, 0, 0, 0]}
    nodal_loads_2 = {1: [200, 0, 0, 0, 0, 0], 2: [0, 200, 0, 0, 0, 0]}
    nodal_loads_3 = {1: [-100, 0, 0, 0, 0, 0], 2: [0, -100, 0, 0, 0, 0]}
    nodal_loads_0 = {}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads_0)
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_1)
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_2)
    (u4, r4) = fcn(node_coords, elements, boundary_conditions, nodal_loads_3)
    assert np.allclose(u1, 0)
    assert np.allclose(r1, 0)
    assert np.allclose(u3, 2 * u2)
    assert np.allclose(r3, 2 * r2)
    assert np.allclose(u4, -u2)
    assert np.allclose(r4, -r2)
    assert np.allclose(np.sum(r1), 0, atol=1e-06)