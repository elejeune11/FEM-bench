def test_simple_beam_discretized_axis_111(fcn):
    """Verify cantilever beam along [1,1,1] axis with tip load against analytical solution."""
    import numpy as np
    L = 10.0
    n_elements = 10
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    Iy = Iz = 8.33e-06
    J = 1e-05
    P = 1000.0
    dx = L / n_elements / np.sqrt(3)
    nodes = np.array([[i * dx, i * dx, i * dx] for i in range(n_elements + 1)])
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {n_elements: [0, -P / np.sqrt(2), P / np.sqrt(2), 0, 0, 0]}
    (u, r) = fcn(nodes, elements, boundary_conditions, nodal_loads)
    L_total = L
    I = Iy
    delta_analytical = P * L_total ** 3 / (3 * E * I)
    tip_index = 6 * n_elements
    delta_y = u[tip_index + 1]
    delta_z = u[tip_index + 2]
    delta_numerical = np.sqrt(delta_y ** 2 + delta_z ** 2)
    assert np.abs(delta_numerical - delta_analytical) / delta_analytical < 0.05

def test_complex_geometry_and_basic_loading(fcn):
    """Test 3D frame analysis with non-trivial geometry under various loading conditions."""
    import numpy as np
    nodes = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [2.0, 0.0, 3.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0001, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0001, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u1, r1) = fcn(nodes, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u1, 0)
    assert np.allclose(r1, 0)
    F = 1000.0
    M = 1000.0
    nodal_loads = {2: [F, F, F, M, M, M]}
    (u2, r2) = fcn(nodes, elements, boundary_conditions, nodal_loads)
    nodal_loads_double = {2: [2 * F, 2 * F, 2 * F, 2 * M, 2 * M, 2 * M]}
    (u3, r3) = fcn(nodes, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u3, 2 * u2)
    assert np.allclose(r3, 2 * r2)
    nodal_loads_neg = {2: [-F, -F, -F, -M, -M, -M]}
    (u4, r4) = fcn(nodes, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u4, -u2)
    assert np.allclose(r4, -r2)
    forces = r2[:3]
    moments = r2[3:6]
    applied_force = np.array([F, F, F])
    applied_moment = np.array([M, M, M])
    assert np.allclose(forces + applied_force, 0)
    r_2 = nodes[2]
    moment_from_force = np.cross(r_2, applied_force)
    total_moment = moments + moment_from_force + applied_moment
    assert np.allclose(total_moment, 0, atol=1e-06)