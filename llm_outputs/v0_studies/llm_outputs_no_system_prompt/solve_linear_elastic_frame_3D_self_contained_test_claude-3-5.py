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
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': 2 * I, 'local_z': None})
    bcs = {0: [1, 1, 1, 1, 1, 1]}
    F = 1000.0
    F_perp = F / np.sqrt(2)
    loads = {n_elem: [F_perp, -F_perp, 0, 0, 0, 0]}
    (u, r) = fcn(nodes, elements, bcs, loads)
    delta_analytical = F * L ** 3 / (3 * E * I)
    delta_perp = delta_analytical / np.sqrt(2)
    tip_ux = u[6 * n_elem + 0]
    tip_uy = u[6 * n_elem + 1]
    assert np.abs(tip_ux - delta_perp) / delta_perp < 0.05
    assert np.abs(tip_uy + delta_perp) / delta_perp < 0.05

def test_complex_geometry_and_basic_loading(fcn):
    """Test 3D frame analysis with non-trivial geometry under various loads"""
    import numpy as np
    nodes = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [3.0, 0.0, 2.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': np.array([0.0, 1.0, 0.0])}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': np.array([0.0, 1.0, 0.0])}]
    bcs = {0: [1, 1, 1, 1, 1, 1]}
    (u0, r0) = fcn(nodes, elements, bcs, {})
    assert np.allclose(u0, 0)
    assert np.allclose(r0, 0)
    loads = {2: [1000.0, 0.0, -2000.0, 0.0, 0.0, 1000.0]}
    (u1, r1) = fcn(nodes, elements, bcs, loads)
    assert not np.allclose(u1, 0)
    assert not np.allclose(r1, 0)
    loads2 = {2: [2000.0, 0.0, -4000.0, 0.0, 0.0, 2000.0]}
    (u2, r2) = fcn(nodes, elements, bcs, loads2)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    loads_neg = {2: [-1000.0, 0.0, 2000.0, 0.0, 0.0, -1000.0]}
    (u_neg, r_neg) = fcn(nodes, elements, bcs, loads_neg)
    assert np.allclose(u_neg, -u1)
    assert np.allclose(r_neg, -r1)
    F = np.array([1000.0, 0.0, -2000.0, 0.0, 0.0, 1000.0])
    R = r1[:6]
    assert np.allclose(F + R[:3], 0)
    assert np.allclose(F[3:] + R[3:], 0)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test ValueError raised for ill-conditioned system due to insufficient constraints"""
    import numpy as np
    import pytest
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': np.array([0.0, 1.0, 0.0])}]
    bcs = {0: [1, 0, 0, 0, 0, 0]}
    loads = {1: [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    with pytest.raises(ValueError):
        fcn(nodes, elements, bcs, loads)