def test_simple_beam_discretized_axis_111(fcn):
    """Verify cantilever beam along [1,1,1] with tip load against analytical solution."""
    L = 10.0
    n_elem = 10
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I = 8.333e-06
    P = 1000.0
    unit_dir = np.array([1, 1, 1]) / np.sqrt(3)
    node_coords = np.array([i * L / n_elem * unit_dir for i in range(n_elem + 1)])
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': 2 * I, 'local_z': np.array([0, -1, 1]) / np.sqrt(2)})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    load_dir = np.array([-1, 1, 0]) / np.sqrt(2)
    nodal_loads = {n_elem: list(P * load_dir) + [0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    delta_analytical = P * L ** 3 / (3 * E * I)
    tip_dof_indices = np.array([6 * n_elem + i for i in range(3)])
    tip_displacement = u[tip_dof_indices]
    delta_numerical = np.dot(tip_displacement, load_dir)
    assert_allclose(delta_numerical, delta_analytical, rtol=0.05)

def test_complex_geometry_and_basic_loading(fcn):
    """Test 3D frame analysis with various loading conditions."""
    node_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [2.0, 0.0, 3.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([1.0, 0.0, 0.0])}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0.0, 1.0, 0.0])}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert_allclose(u0, np.zeros_like(u0))
    assert_allclose(r0, np.zeros_like(r0))
    F = 1000.0
    M = 2000.0
    nodal_loads = {2: [F, 0.0, -F, M, 0.0, 0.0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    nodal_loads_2x = {2: [2 * F, 0.0, -2 * F, 2 * M, 0.0, 0.0]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_2x)
    assert_allclose(u2, 2 * u1, rtol=1e-10)
    assert_allclose(r2, 2 * r1, rtol=1e-10)
    nodal_loads_neg = {2: [-F, 0.0, F, -M, 0.0, 0.0]}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert_allclose(u3, -u1, rtol=1e-10)
    assert_allclose(r3, -r1, rtol=1e-10)
    forces = r1[:3]
    moments = r1[3:6]
    applied_force = np.array([F, 0, -F])
    applied_moment = np.array([M, 0, 0])
    moment_arm = node_coords[2] - node_coords[0]
    total_moment = np.cross(moment_arm, applied_force) + applied_moment
    assert_allclose(forces, -applied_force, rtol=1e-10)
    assert_allclose(moments, -total_moment, rtol=1e-10)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test ValueError raised for under-constrained structure."""
    node_coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0.0, 1.0, 0.0])}]
    boundary_conditions = {0: [1, 0, 0, 0, 0, 0]}
    nodal_loads = {1: [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    with pytest.raises(ValueError):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)