def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with respect to known analytical solution.
    Cantilever beam along [1,1,1] axis with 10 elements and tip load.
    Checks tip deflection against analytical solution.
    """
    import numpy as np
    L = np.sqrt(3)
    n_elem = 10
    dx = L / n_elem / np.sqrt(3)
    nodes = np.array([[i * dx, i * dx, i * dx] for i in range(n_elem + 1)])
    E = 200000000000.0
    nu = 0.3
    d = 0.1
    A = np.pi * d ** 2 / 4
    I = np.pi * d ** 4 / 64
    J = 2 * I
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': None})
    bcs = {0: [1, 1, 1, 1, 1, 1]}
    F = 1000.0
    F_comp = F / np.sqrt(2)
    loads = {n_elem: [0, -F_comp, -F_comp, 0, 0, 0]}
    (u, r) = fcn(nodes, elements, bcs, loads)
    delta_analytical = F * L ** 3 / (3 * E * I)
    delta_comp = delta_analytical / np.sqrt(2)
    tip_index = 6 * n_elem
    uy = u[tip_index + 1]
    uz = u[tip_index + 2]
    delta_numerical = np.sqrt(uy ** 2 + uz ** 2)
    assert np.abs(delta_numerical - delta_analytical) / delta_analytical < 0.05

def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on non-trivial geometry under various loads.
    Verifies zero loads, linearity, sign reversal, and static equilibrium.
    """
    import numpy as np
    nodes = np.array([[0, 0, 0], [0, 0, 2], [1.5, 0, 2]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': None}]
    bcs = {0: [1, 1, 1, 1, 1, 1]}
    loads_zero = {}
    (u0, r0) = fcn(nodes, elements, bcs, loads_zero)
    assert np.allclose(u0, 0)
    assert np.allclose(r0, 0)
    P = 1000.0
    M = 2000.0
    loads_1 = {2: [P, 0, -P, M, 0, M]}
    (u1, r1) = fcn(nodes, elements, bcs, loads_1)
    loads_2 = {2: [2 * P, 0, -2 * P, 2 * M, 0, 2 * M]}
    (u2, r2) = fcn(nodes, elements, bcs, loads_2)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    loads_neg = {2: [-P, 0, P, -M, 0, -M]}
    (u_neg, r_neg) = fcn(nodes, elements, bcs, loads_neg)
    assert np.allclose(u_neg, -u1)
    assert np.allclose(r_neg, -r1)
    F_sum = np.zeros(3)
    F_sum += np.array([P, 0, -P])
    F_sum += np.array([r1[0], r1[1], r1[2]])
    assert np.allclose(F_sum, 0)
    M_sum = np.zeros(3)
    M_sum += np.array([M, 0, M])
    M_sum += np.array([r1[3], r1[4], r1[5]])
    M_sum += np.cross([0, 0, 2], [P, 0, -P])
    M_sum += np.cross([1.5, 0, 2], [P, 0, -P])
    assert np.allclose(M_sum, 0)