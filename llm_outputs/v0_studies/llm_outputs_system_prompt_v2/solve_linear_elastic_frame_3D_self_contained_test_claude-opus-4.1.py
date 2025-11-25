def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L_total = 10.0
    n_elements = 10
    L_elem = L_total / n_elements
    direction = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    node_coords = np.array([i * L_elem * direction for i in range(n_elements + 1)])
    E = 200000000000.0
    nu = 0.3
    d = 0.1
    A = np.pi * d ** 2 / 4
    I = np.pi * d ** 4 / 64
    J = 2 * I
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0, 0, 1])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    force_perpendicular = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    F = 1000.0
    nodal_loads = {n_elements: list(F * force_perpendicular) + [0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_disp = u[-6:-3]
    disp_magnitude = np.linalg.norm(tip_disp)
    analytical_deflection = F * L_total ** 3 / (3 * E * I)
    assert np.abs(disp_magnitude - analytical_deflection) / analytical_deflection < 0.01
    total_reaction_force = r[0:3]
    total_reaction_moment = r[3:6]
    assert np.allclose(total_reaction_force, -F * force_perpendicular, rtol=1e-06)
    moment_arm = L_total * direction
    expected_moment = np.cross(moment_arm, F * force_perpendicular)
    assert np.allclose(total_reaction_moment, -expected_moment, rtol=1e-06)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 3.0, 0.0], [0.0, 3.0, 0.0], [1.0, 1.5, 2.0], [3.0, 1.5, 1.0]])
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.33e-06
    I_z = 8.33e-06
    J = 1.66e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 3, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 0, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 2, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 5, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 3, 'node_j': 5, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 1: [1, 1, 1, 0, 0, 0], 3: [0, 1, 1, 0, 0, 0]}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, {})
    assert np.allclose(u_zero, 0, atol=1e-12)
    assert np.allclose(r_zero, 0, atol=1e-12)
    base_loads = {2: [1000.0, -500.0, 200.0, 50.0, -30.0, 20.0], 4: [-800.0, 600.0, -300.0, -40.0, 25.0, -15.0], 5: [500.0, -200.0, 100.0, 30.0, -20.0, 10.0]}
    (u_base, r_base) = fcn(node_coords, elements, boundary_conditions, base_loads)
    assert not np.allclose(u_base, 0)
    assert not np.allclose(r_base, 0)
    double_loads = {k: [2 * v_i for v_i in v] for (k, v) in base_loads.items()}
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, double_loads)
    assert np.allclose(u_double, 2 * u_base, rtol=1e-10)
    assert np.allclose(r_double, 2 * r_base, rtol=1e-10)
    neg_loads = {k: [-v_i for v_i in v] for (k, v) in base_loads.items()}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, neg_loads)
    assert np.allclose(u_neg, -u_base, rtol=1e-10)
    assert np.allclose(r_neg, -r_base, rtol=1e-10)
    total_applied_force = np.zeros(3)
    total_applied_moment = np.zeros(3)
    for (node_idx, loads) in base_loads.items():
        total_applied_force += np.array(loads[:3])
        total_applied_moment += np.array(loads[3:])
        total_applied_moment += np.cross(node_coords[node_idx], loads[:3])
    total_reaction_force = np.zeros(3)
    total_reaction_moment = np.zeros(3)
    for node_idx in range(len(node_coords)):
        node_reactions = r_base[6 * node_idx:6 * (node_idx + 1)]
        total_reaction_force += node_reactions[:3]
        total_reaction_moment += node_reactions[3:]
        total_reaction_moment += np.cross(node_coords[node_idx], node_reactions[:3])
    assert np.allclose(total_applied_force + total_reaction_force, 0, atol=1e-08)
    assert np.allclose(total_applied_moment + total_reaction_moment, 0, atol=1e-08)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff).
    The solver should detect this (cond(K_ff) >= 1e16) and raise a ValueError."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-06
    I_z = 1e-06
    J = 2e-06
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}]
    boundary_conditions = {}
    nodal_loads = {1: [0, 0, 1000.0, 0, 0, 0]}
    with pytest.raises(ValueError):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)