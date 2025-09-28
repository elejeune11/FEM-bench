def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    direction = np.array([1, 1, 1]) / np.sqrt(3)
    n_elements = 10
    n_nodes = n_elements + 1
    node_coords = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        node_coords[i] = i * (L / n_elements) * direction
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = I_z = 8.333e-06
    J = 1.667e-05
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0, 0, 1])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F = 1000.0
    tip_force = np.array([0, F, 0, 0, 0, 0])
    nodal_loads = {n_nodes - 1: tip_force}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    analytical_deflection = F * L ** 3 / (3 * E * I_y)
    tip_node_dofs = (n_nodes - 1) * 6
    computed_deflection = u[tip_node_dofs + 1]
    assert abs(computed_deflection - analytical_deflection) / analytical_deflection < 0.05
    reaction_force_y = r[1]
    assert abs(reaction_force_y + F) / F < 0.01
    reaction_moment_z = r[5]
    expected_moment = F * L
    assert abs(reaction_moment_z + expected_moment) / expected_moment < 0.05

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 3.0]])
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = I_z = 8.333e-06
    J = 1.667e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0, 0, 1])}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0, 0, 1])}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([1, 0, 0])}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u1, 0.0, atol=1e-12)
    assert np.allclose(r1, 0.0, atol=1e-12)
    nodal_loads = {2: [100.0, 200.0, 0.0, 0.0, 0.0, 500.0], 3: [0.0, 0.0, -300.0, 1000.0, 0.0, 0.0]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.any(np.abs(u2) > 1e-10)
    assert np.any(np.abs(r2) > 1e-10)
    nodal_loads_double = {2: [200.0, 400.0, 0.0, 0.0, 0.0, 1000.0], 3: [0.0, 0.0, -600.0, 2000.0, 0.0, 0.0]}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u3, 2.0 * u2, rtol=1e-12)
    assert np.allclose(r3, 2.0 * r2, rtol=1e-12)
    nodal_loads_neg = {2: [-100.0, -200.0, 0.0, 0.0, 0.0, -500.0], 3: [0.0, 0.0, 300.0, -1000.0, 0.0, 0.0]}
    (u4, r4) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u4, -u2, rtol=1e-12)
    assert np.allclose(r4, -r2, rtol=1e-12)
    applied_forces = np.array([100.0, 200.0, -300.0])
    applied_moments = np.array([1000.0, 0.0, 500.0])
    reaction_forces = r2[0:3]
    reaction_moments = r2[3:6]
    total_forces = applied_forces + reaction_forces
    total_moments = applied_moments + reaction_moments
    assert np.allclose(total_forces, 0.0, atol=1e-08)
    assert np.allclose(total_moments, 0.0, atol=1e-06)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff).
    The solver should detect this (cond(K_ff) >= 1e16) and raise a ValueError."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = I_z = 8.333e-06
    J = 1.667e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0, 0, 1])}]
    boundary_conditions = {}
    nodal_loads = {1: [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    with pytest.raises(ValueError):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)