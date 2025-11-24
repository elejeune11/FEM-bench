def test_simple_beam_discretized_axis_111(fcn):
    """Verify cantilever beam along [1,1,1] with tip load against analytical solution."""
    L = 10.0
    n_elem = 10
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    Iy = Iz = 8.333e-06
    J = 1.407e-05
    P = 1000.0
    unit_dir = np.array([1, 1, 1]) / np.sqrt(3)
    dx = L / n_elem * unit_dir
    nodes = np.array([i * dx for i in range(n_elem + 1)])
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': np.array([0, -1, 1]) / np.sqrt(2)})
    bc = {0: [1, 1, 1, 1, 1, 1]}
    loads = {n_elem: [0, -P / np.sqrt(2), P / np.sqrt(2), 0, 0, 0]}
    (u, r) = fcn(nodes, elements, bc, loads)
    delta_analytical = P * L ** 3 / (3 * E * Iy)
    tip_disp = np.sqrt(u[6 * n_elem + 1] ** 2 + u[6 * n_elem + 2] ** 2)
    assert_allclose(tip_disp, delta_analytical, rtol=0.05)

def test_complex_geometry_and_basic_loading(fcn):
    """Test 3D frame analysis with various loading conditions."""
    nodes = np.array([[0, 0, 0], [3, 0, 0], [3, 2, 0], [0, 2, 0], [0, 0, 2], [3, 0, 2]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0, 0, 1])}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0, 0, 1])}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0, 0, 1])}, {'node_i': 3, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0, 0, 1])}, {'node_i': 0, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0, 1, 0])}, {'node_i': 1, 'node_j': 5, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0, 1, 0])}]
    bc = {0: [1, 1, 1, 1, 1, 1], 3: [1, 1, 1, 1, 1, 1]}
    (u1, r1) = fcn(nodes, elements, bc, {})
    assert_allclose(u1, np.zeros_like(u1))
    assert_allclose(r1, np.zeros_like(r1))
    loads = {2: [1000, 0, -2000, 500, 0, 0]}
    (u2, r2) = fcn(nodes, elements, bc, loads)
    assert not np.allclose(u2, np.zeros_like(u2))
    loads_double = {2: [2000, 0, -4000, 1000, 0, 0]}
    (u3, r3) = fcn(nodes, elements, bc, loads_double)
    assert_allclose(u3, 2 * u2)
    assert_allclose(r3, 2 * r2)
    loads_neg = {2: [-1000, 0, 2000, -500, 0, 0]}
    (u4, r4) = fcn(nodes, elements, bc, loads_neg)
    assert_allclose(u4, -u2)
    assert_allclose(r4, -r2)
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    for node in range(len(nodes)):
        force = r2[6 * node:6 * node + 3]
        moment = r2[6 * node + 3:6 * node + 6]
        if node in loads:
            force += loads[node][:3]
            moment += loads[node][3:]
        total_force += force
        total_moment += moment + np.cross(nodes[node], force)
    assert_allclose(total_force, np.zeros(3), atol=1e-06)
    assert_allclose(total_moment, np.zeros(3), atol=1e-06)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test ValueError raised for ill-conditioned system."""
    nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0, 0, 1])}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0, 0, 1])}]
    bc = {0: [1, 0, 1, 0, 0, 0]}
    loads = {2: [1000, 0, 0, 0, 0, 0]}
    with pytest.raises(ValueError):
        fcn(nodes, elements, bc, loads)