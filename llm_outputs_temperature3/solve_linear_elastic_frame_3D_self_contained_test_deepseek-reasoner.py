def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    E = 200000000000.0
    I = 1e-06
    A = 0.001
    nu = 0.3
    J = 2e-06
    n_elements = 10
    nodes = []
    for i in range(n_elements + 1):
        s = i * L / n_elements
        nodes.append([s, s, s])
    node_coords = np.array(nodes)
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_mag = 1000.0
    F_dir = np.array([1, -1, 0])
    F_dir = F_dir / np.linalg.norm(F_dir)
    force = F_mag * F_dir
    nodal_loads = {n_elements: [force[0], force[1], force[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_disp = u[6 * n_elements:6 * n_elements + 3]
    measured_deflection = np.dot(tip_disp, F_dir)
    analytical_deflection = F_mag * L ** 3 / (3 * E * I)
    assert abs(measured_deflection - analytical_deflection) / analytical_deflection < 0.01

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions."""
    node_coords = np.array([[0, 0, 0], [5, 0, 0], [5, 4, 0], [5, 4, 3], [2, 4, 3]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': None}, {'node_i': 3, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0.0, atol=1e-12)
    assert np.allclose(r_zero, 0.0, atol=1e-12)
    nodal_loads_base = {1: [1000, 500, -200, 0, 0, 0], 2: [0, 0, 0, 50, -30, 20], 4: [-200, 300, 100, 0, 10, -5]}
    (u_base, r_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads_base)
    assert not np.allclose(u_base[6:12], 0.0)
    assert not np.allclose(u_base[12:18], 0.0)
    assert not np.allclose(u_base[24:30], 0.0)
    assert not np.allclose(r_base[0:6], 0.0)
    nodal_loads_double = {node: [2 * x for x in loads] for (node, loads) in nodal_loads_base.items()}
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u_double, 2 * u_base, rtol=1e-10)
    assert np.allclose(r_double, 2 * r_base, rtol=1e-10)
    nodal_loads_neg = {node: [-x for x in loads] for (node, loads) in nodal_loads_base.items()}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u_neg, -u_base, rtol=1e-10)
    assert np.allclose(r_neg, -r_base, rtol=1e-10)
    total_force_applied = np.zeros(3)
    total_moment_applied = np.zeros(3)
    for (node, loads) in nodal_loads_base.items():
        force = np.array(loads[0:3])
        moment = np.array(loads[3:6])
        pos = node_coords[node]
        total_force_applied += force
        total_moment_applied += moment + np.cross(pos, force)
    total_force_react = np.zeros(3)
    total_moment_react = np.zeros(3)
    for node in boundary_conditions:
        if boundary_conditions[node][0]:
            react_force = np.array(r_base[6 * node:6 * node + 3])
            react_moment = np.array(r_base[6 * node + 3:6 * node + 6])
            pos = node_coords[node]
            total_force_react += react_force
            total_moment_react += react_moment + np.cross(pos, react_force)
    assert np.allclose(total_force_applied + total_force_react, 0.0, atol=1e-10)
    assert np.allclose(total_moment_applied + total_moment_react, 0.0, atol=1e-10)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff)."""
    node_coords = np.array([[0, 0, 0], [5, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': None}]
    boundary_conditions = {}
    nodal_loads = {0: [1000, 0, 0, 0, 0, 0]}
    with pytest.raises(ValueError):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)