def test_simple_beam_discretized_axis_111(fcn):
    """Verify cantilever beam along [1,1,1] axis with tip load against analytical solution"""
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
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0, -1 / np.sqrt(2), 1 / np.sqrt(2)])})
    bcs = {0: [1, 1, 1, 1, 1, 1]}
    P = 1000.0
    F = P / np.sqrt(6) * np.array([-2, 1, 1])
    loads = {n_elem: [*F, 0, 0, 0]}
    (u, r) = fcn(nodes, elements, bcs, loads)
    delta_analytical = P * L ** 3 / (3 * E * I)
    tip_disp = u[6 * n_elem:6 * n_elem + 3]
    delta_numerical = np.dot(tip_disp, F) / P
    assert np.abs(delta_numerical - delta_analytical) / delta_analytical < 0.01

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on non-trivial geometry under various loads"""
    import numpy as np
    nodes = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]])
    element = {'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': None}
    elements = [{**element, 'node_i': 0, 'node_j': 1}, {**element, 'node_i': 1, 'node_j': 2}]
    bcs = {0: [1, 1, 1, 1, 1, 1]}
    (u0, r0) = fcn(nodes, elements, bcs, {})
    assert np.allclose(u0, 0)
    assert np.allclose(r0, 0)
    P = 1000.0
    loads = {2: [P, 0, 0, 0, 0, P]}
    (u1, r1) = fcn(nodes, elements, bcs, loads)
    assert not np.allclose(u1, 0)
    assert not np.allclose(r1, 0)
    loads2 = {2: [2 * P, 0, 0, 0, 0, 2 * P]}
    (u2, r2) = fcn(nodes, elements, bcs, loads2)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    loads3 = {2: [-P, 0, 0, 0, 0, -P]}
    (u3, r3) = fcn(nodes, elements, bcs, loads3)
    assert np.allclose(u3, -u1)
    assert np.allclose(r3, -r1)
    F = r1[:3]
    M = r1[3:6]
    F_app = np.array([P, 0, 0])
    M_app = np.array([0, 0, P])
    r_pos = np.array([0, 1, 1])
    assert np.allclose(F + F_app, 0)
    assert np.allclose(M + M_app + np.cross(r_pos, F_app), 0)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test ValueError raised for ill-conditioned system due to insufficient constraints"""
    import numpy as np
    import pytest
    nodes = np.array([[0, 0, 0], [1, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': None}]
    bcs = {0: [1, 0, 0, 0, 0, 0], 1: [1, 0, 0, 0, 0, 0]}
    loads = {1: [0, 1000, 0, 0, 0, 0]}
    with pytest.raises(ValueError):
        fcn(nodes, elements, bcs, loads)