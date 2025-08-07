def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to known analytical solution for cantilever beam along [1,1,1] axis."""
    import numpy as np
    L = 10.0
    n_elem = 10
    dx = L / n_elem / np.sqrt(3)
    nodes = np.array([[i * dx, i * dx, i * dx] for i in range(n_elem + 1)])
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    Iy = Iz = 8.33e-06
    J = 1e-05
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J})
    bc = {0: [1, 1, 1, 1, 1, 1]}
    F = 1000.0
    loads = {n_elem: [0, -F / np.sqrt(2), F / np.sqrt(2), 0, 0, 0]}
    L_total = L / np.sqrt(3)
    delta_analytical = F * L_total ** 3 / (3 * E * Iy)
    (u, r) = fcn(nodes, elements, bc, loads)
    tip_idx = 6 * n_elem + np.array([1, 2])
    delta_numerical = u[tip_idx]
    delta_magnitude = np.sqrt(np.sum(delta_numerical ** 2))
    assert np.abs(delta_magnitude - delta_analytical) / delta_analytical < 0.05

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on non-trivial geometry under various loads."""
    import numpy as np
    nodes = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}]
    bc = {0: [1, 1, 1, 1, 1, 1]}
    (u0, r0) = fcn(nodes, elements, bc, {})
    assert np.allclose(u0, 0)
    assert np.allclose(r0, 0)
    F = 1000.0
    M = 1000.0
    loads = {2: [F, 0, 0, 0, 0, M]}
    (u1, r1) = fcn(nodes, elements, bc, loads)
    assert not np.allclose(u1, 0)
    assert not np.allclose(r1, 0)
    loads2 = {2: [2 * F, 0, 0, 0, 0, 2 * M]}
    (u2, r2) = fcn(nodes, elements, bc, loads2)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    loads3 = {2: [-F, 0, 0, 0, 0, -M]}
    (u3, r3) = fcn(nodes, elements, bc, loads3)
    assert np.allclose(u3, -u1)
    assert np.allclose(r3, -r1)
    forces = r1.reshape(-1, 6)
    total_force = np.sum(forces[:, :3], axis=0)
    total_moment = np.sum(forces[:, 3:], axis=0)
    assert np.allclose(total_force, 0, atol=1e-06)
    assert np.allclose(total_moment, 0, atol=1e-06)

def test_condition_number(fcn):
    """Test effect of boundary conditions on stiffness matrix condition number."""
    import numpy as np
    nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}]
    loads = {2: [1000, 0, 0, 0, 0, 1000]}
    (u1, r1) = fcn(nodes, elements, {}, loads)
    assert np.allclose(u1, 0)
    assert np.allclose(r1, 0)
    bc = {0: [1, 1, 1, 1, 1, 1]}
    (u2, r2) = fcn(nodes, elements, bc, loads)
    assert not np.allclose(u2, 0)
    assert not np.allclose(r2, 0)