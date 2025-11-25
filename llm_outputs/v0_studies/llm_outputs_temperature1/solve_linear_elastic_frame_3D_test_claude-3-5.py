def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with respect to known analytical solution.
    Cantilever beam along [1,1,1] axis with tip load perpendicular to beam axis.
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
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F = 1000.0
    F_perp = F / np.sqrt(2)
    nodal_loads = {n_elem: [F_perp, -F_perp, 0, 0, 0, 0]}
    (u, r) = fcn(nodes, elements, boundary_conditions, nodal_loads)
    delta_analytical = F * L ** 3 / (3 * E * I)
    delta_perp = np.sqrt(u[6 * n_elem] ** 2 + u[6 * n_elem + 1] ** 2)
    assert np.isclose(delta_perp, delta_analytical, rtol=0.05)

def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on non-trivial geometry under various loads.
    Verifies linearity properties and static equilibrium.
    """
    import numpy as np
    nodes = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [1.0, 1.0, 2.0]])
    props = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    elements = [{**props, 'node_i': 0, 'node_j': 1, 'local_z': None}, {**props, 'node_i': 1, 'node_j': 2, 'local_z': None}, {**props, 'node_i': 2, 'node_j': 3, 'local_z': None}]
    bcs = {0: [1, 1, 1, 1, 1, 1]}
    loads_zero = {}
    (u0, r0) = fcn(nodes, elements, bcs, loads_zero)
    assert np.allclose(u0, 0)
    assert np.allclose(r0, 0)
    F = 1000.0
    M = 1000.0
    loads = {3: [F, F, F, M, M, M]}
    (u1, r1) = fcn(nodes, elements, bcs, loads)
    assert not np.allclose(u1, 0)
    assert not np.allclose(r1, 0)
    loads_2x = {3: [2 * F, 2 * F, 2 * F, 2 * M, 2 * M, 2 * M]}
    (u2, r2) = fcn(nodes, elements, bcs, loads_2x)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    loads_neg = {3: [-F, -F, -F, -M, -M, -M]}
    (u3, r3) = fcn(nodes, elements, bcs, loads_neg)
    assert np.allclose(u3, -u1)
    assert np.allclose(r3, -r1)
    F_sum = r1[:3] + loads[3][:3]
    assert np.allclose(F_sum, 0)
    M_sum = r1[3:6]
    for (node, load) in loads.items():
        r = nodes[node]
        F = load[:3]
        M = load[3:6]
        M_sum += np.cross(r, F) + M
    assert np.allclose(M_sum, 0, atol=1e-06)