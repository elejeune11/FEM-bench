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
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([1, -1, 0]) / np.sqrt(2)})
    bcs = {0: [1, 1, 1, 1, 1, 1]}
    F = 1000.0
    F_dir = np.array([1, -1, 0]) / np.sqrt(2)
    loads = {n_elem: list(F * F_dir) + [0, 0, 0]}
    (u, r) = fcn(nodes, elements, bcs, loads)
    delta_analytical = F * L ** 3 / (3 * E * I)
    tip_disp = u[6 * n_elem:6 * n_elem + 3]
    delta_numerical = np.dot(tip_disp, F_dir)
    assert np.abs(delta_numerical - delta_analytical) / delta_analytical < 0.05

def test_complex_geometry_and_basic_loading(fcn):
    """Test 3D frame analysis with various loading conditions"""
    import numpy as np
    nodes = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [3.0, 0.0, 2.0]])
    props = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    elements = [{**props, 'node_i': 0, 'node_j': 1, 'local_z': np.array([1, 0, 0])}, {**props, 'node_i': 1, 'node_j': 2, 'local_z': np.array([0, 1, 0])}]
    bcs = {0: [1, 1, 1, 1, 1, 1]}
    (u0, r0) = fcn(nodes, elements, bcs, {})
    assert np.allclose(u0, 0)
    assert np.allclose(r0, 0)
    F = 1000.0
    loads = {2: [0, 0, -F, 0, 0, 0]}
    (u1, r1) = fcn(nodes, elements, bcs, loads)
    assert not np.allclose(u1, 0)
    assert not np.allclose(r1, 0)
    loads2 = {2: [0, 0, -2 * F, 0, 0, 0]}
    (u2, r2) = fcn(nodes, elements, bcs, loads2)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    loads3 = {2: [0, 0, F, 0, 0, 0]}
    (u3, r3) = fcn(nodes, elements, bcs, loads3)
    assert np.allclose(u3, -u1)
    assert np.allclose(r3, -r1)
    forces = r1[:3]
    moments = r1[3:6]
    assert np.allclose(forces + [0, 0, -F], 0)
    assert np.allclose(moments + np.cross([0, 0, 2], [0, 0, -F]) + np.cross([3, 0, 2], [0, 0, -F]), 0)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test ValueError raised for ill-conditioned system due to insufficient constraints"""
    import numpy as np
    import pytest
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': np.array([0, 1, 0])}]
    bcs = {0: [1, 0, 0, 0, 0, 0], 1: [1, 0, 0, 0, 0, 0]}
    loads = {1: [0, 1000, 0, 0, 0, 0]}
    with pytest.raises(ValueError):
        fcn(nodes, elements, bcs, loads)