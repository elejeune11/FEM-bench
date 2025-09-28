def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L_total = 10.0
    n_elements = 10
    L_elem = L_total / n_elements
    E = 200000000000.0
    nu = 0.3
    d = 0.1
    A = np.pi * d ** 2 / 4
    I = np.pi * d ** 4 / 64
    J = np.pi * d ** 4 / 32
    direction = np.array([1, 1, 1]) / np.sqrt(3)
    node_coords = np.array([i * L_elem * direction for i in range(n_elements + 1)])
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0, 0, 1])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_magnitude = 1000.0
    force_direction = np.array([1, 1, -2]) / np.sqrt(6)
    nodal_loads = {n_elements: [F_magnitude * force_direction[0], F_magnitude * force_direction[1], F_magnitude * force_direction[2], 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_node_idx = n_elements
    tip_disp = u[6 * tip_node_idx:6 * tip_node_idx + 3]
    delta_analytical = F_magnitude * L_total ** 3 / (3 * E * I)
    tip_disp_magnitude = np.linalg.norm(tip_disp)
    assert tip_disp_magnitude > 0
    assert np.abs(tip_disp_magnitude - delta_analytical) / delta_analytical < 0.01

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0, 0, 0], [2, 0, 0], [2, 3, 0], [2, 0, 2]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0, 0, 1])}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0, 0, 1])}, {'node_i': 1, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([1, 0, 0])}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u, 0)
    assert np.allclose(r, 0)
    base_loads = {2: [100, 200, -50, 10, 20, 30], 3: [-150, 100, 200, -15, 25, -10]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, base_loads)
    assert not np.allclose(u1, 0)
    assert not np.allclose(r1, 0)
    double_loads = {k: [2 * v_i for v_i in v] for (k, v) in base_loads.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, double_loads)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    neg_loads = {k: [-v_i for v_i in v] for (k, v) in base_loads.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, neg_loads)
    assert np.allclose(u3, -u1)
    assert np.allclose(r3, -r1)
    total_applied_force = np.zeros(3)
    total_applied_moment = np.zeros(3)
    for (node_idx, loads) in base_loads.items():
        total_applied_force += loads[:3]
        total_applied_moment += loads[3:]
        total_applied_moment += np.cross(node_coords[node_idx], loads[:3])
    reaction_force = r1[:3]
    reaction_moment = r1[3:6]
    assert np.allclose(reaction_force + total_applied_force, 0, atol=1e-08)
    assert np.allclose(reaction_moment + total_applied_moment, 0, atol=1e-08)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff).
    The solver should detect this (cond(K_ff) >= 1e16) and raise a ValueError."""
    node_coords = np.array([[0, 0, 0], [1, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': None}]
    boundary_conditions = {}
    nodal_loads = {1: [100, 0, 0, 0, 0, 0]}
    with pytest.raises(ValueError):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)