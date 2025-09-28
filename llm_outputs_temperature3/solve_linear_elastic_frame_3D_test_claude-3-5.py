def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification against analytical solution for cantilever beam along [1,1,1] axis.
    Ten equal 3D frame elements with tip load perpendicular to beam axis.
    Checks tip deflection against analytical reference solution.
    """
    import numpy as np
    L = np.sqrt(3)
    n_elem = 10
    dx = L / n_elem / np.sqrt(3)
    nodes = np.array([[i * dx, i * dx, i * dx] for i in range(n_elem + 1)])
    E = 200000000000.0
    nu = 0.3
    d = 0.01
    A = np.pi * d ** 2 / 4
    I = np.pi * d ** 4 / 64
    J = 2 * I
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': None})
    bcs = {0: [1, 1, 1, 1, 1, 1]}
    F = 100
    F_perp = F / np.sqrt(2)
    loads = {n_elem: [0, F_perp, -F_perp, 0, 0, 0]}
    (u, r) = fcn(nodes, elements, bcs, loads)
    delta_analytical = F * L ** 3 / (3 * E * I)
    tip_y = u[6 * n_elem + 1]
    tip_z = u[6 * n_elem + 2]
    delta_computed = np.sqrt(tip_y ** 2 + tip_z ** 2)
    assert np.abs(delta_computed - delta_analytical) / delta_analytical < 0.05

def test_complex_geometry_and_basic_loading(fcn):
    """
    Tests 3D frame analysis on non-trivial geometry under various loading conditions.
    Verifies zero loads, linear scaling of response, and static equilibrium.
    """
    import numpy as np
    nodes = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.001, 'I_y': 1e-08, 'I_z': 1e-08, 'J': 2e-08, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.001, 'I_y': 1e-08, 'I_z': 1e-08, 'J': 2e-08, 'local_z': None}]
    bcs = {0: [1, 1, 1, 1, 1, 1]}
    loads_zero = {}
    (u0, r0) = fcn(nodes, elements, bcs, loads_zero)
    assert np.allclose(u0, 0)
    assert np.allclose(r0, 0)
    F = 1000
    M = 500
    loads = {2: [F, 0, F, M, 0, M]}
    (u1, r1) = fcn(nodes, elements, bcs, loads)
    loads_2x = {2: [2 * F, 0, 2 * F, 2 * M, 0, 2 * M]}
    (u2, r2) = fcn(nodes, elements, bcs, loads_2x)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    loads_neg = {2: [-F, 0, -F, -M, 0, -M]}
    (u3, r3) = fcn(nodes, elements, bcs, loads_neg)
    assert np.allclose(u3, -u1)
    assert np.allclose(r3, -r1)
    reactions = r1[:6]
    applied = np.array([F, 0, F, M, 0, M])
    assert np.allclose(reactions[:3], -applied[:3])
    M_applied = np.array([M, 0, M])
    r_2 = np.array([0, 1, 1])
    M_force = np.cross(r_2, applied[:3])
    M_total = M_force + M_applied
    assert np.allclose(reactions[3:], -M_total)