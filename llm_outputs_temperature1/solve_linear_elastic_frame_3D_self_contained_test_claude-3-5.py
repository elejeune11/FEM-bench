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
    F = 1000.0
    P = F / np.sqrt(2)
    loads = {n_elem: [0, -P, P, 0, 0, 0]}
    (u, r) = fcn(nodes, elements, bcs, loads)
    delta_analytical = F * L ** 3 / (3 * E * I)
    tip_uy = u[6 * n_elem + 1]
    tip_uz = u[6 * n_elem + 2]
    tip_total = np.sqrt(tip_uy ** 2 + tip_uz ** 2)
    assert np.abs(tip_total - delta_analytical) / delta_analytical < 0.01

def test_complex_geometry_and_basic_loading(fcn):
    """Test 3D frame analysis with various loading conditions"""
    import numpy as np
    nodes = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': np.array([1, 0, 0])}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': np.array([1, 0, 0])}]
    bcs = {0: [1, 1, 1, 1, 1, 1]}
    loads = {}
    (u0, r0) = fcn(nodes, elements, bcs, loads)
    assert np.allclose(u0, 0)
    assert np.allclose(r0, 0)
    F = 1000.0
    M = 1000.0
    loads = {2: [F, 0, 0, 0, 0, M]}
    (u1, r1) = fcn(nodes, elements, bcs, loads)
    assert not np.allclose(u1, 0)
    assert not np.allclose(r1, 0)
    loads = {2: [2 * F, 0, 0, 0, 0, 2 * M]}
    (u2, r2) = fcn(nodes, elements, bcs, loads)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    loads = {2: [-F, 0, 0, 0, 0, -M]}
    (u3, r3) = fcn(nodes, elements, bcs, loads)
    assert np.allclose(u3, -u1)
    assert np.allclose(r3, -r1)
    loads = {2: [F, 0, 0, 0, 0, M]}
    (u, r) = fcn(nodes, elements, bcs, loads)
    F_sum = np.zeros(3)
    M_sum = np.zeros(3)
    F_sum += np.array([F, 0, 0])
    M_sum += np.array([0, 0, M])
    M_sum += np.cross(nodes[2], np.array([F, 0, 0]))
    F_sum += r[:3]
    M_sum += r[3:6]
    assert np.allclose(F_sum, 0, atol=1e-06)
    assert np.allclose(M_sum, 0, atol=1e-06)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test ValueError raised for ill-conditioned system"""
    import numpy as np
    import pytest
    nodes = np.array([[0, 0, 0], [1, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': np.array([0, 0, 1])}]
    bcs = {0: [1, 0, 0, 0, 0, 0], 1: [1, 0, 0, 0, 0, 0]}
    loads = {1: [0, 1000, 0, 0, 0, 0]}
    with pytest.raises(ValueError):
        fcn(nodes, elements, bcs, loads)