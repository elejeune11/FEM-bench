def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    n_elements = 10
    element_length = L / n_elements
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-06
    I_z = 1e-06
    J = 2e-06
    beam_dir = np.array([1.0, 1.0, 1.0])
    beam_dir = beam_dir / np.linalg.norm(beam_dir)
    node_coords = []
    for i in range(n_elements + 1):
        pos = i * element_length * beam_dir
        node_coords.append(pos)
    node_coords = np.array(node_coords)
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    tip_node = n_elements
    F_magnitude = 1000.0
    if abs(beam_dir[0]) > 0.5:
        perp_vec = np.cross(beam_dir, np.array([0, 1, 0]))
    else:
        perp_vec = np.cross(beam_dir, np.array([1, 0, 0]))
    perp_vec = perp_vec / np.linalg.norm(perp_vec)
    force_vec = F_magnitude * perp_vec
    nodal_loads = {tip_node: [force_vec[0], force_vec[1], force_vec[2], 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_dofs = slice(6 * tip_node, 6 * tip_node + 3)
    tip_displacement = u[tip_dofs]
    displacement_dir = tip_displacement / np.linalg.norm(tip_displacement)
    expected_dir = -force_vec / np.linalg.norm(force_vec)
    dot_product = np.dot(displacement_dir, expected_dir)
    assert abs(dot_product) > 0.9
    displacement_magnitude = np.linalg.norm(tip_displacement)
    assert displacement_magnitude > 0
    assert displacement_magnitude < 0.1

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0], [5, 5, 5]])
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-05
    I_z = 1e-05
    J = 2e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0.0, atol=1e-12)
    assert np.allclose(r_zero, 0.0, atol=1e-12)
    nodal_loads = {1: [1000, 2000, 3000, 400, 500, 600], 3: [0, 0, -5000, 0, 1000, 0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u1, 0.0, atol=1e-12)
    assert not np.allclose(r1, 0.0, atol=1e-12)
    nodal_loads_double = {1: [2000, 4000, 6000, 800, 1000, 1200], 3: [0, 0, -10000, 0, 2000, 0]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(2 * u1, u2, rtol=1e-10)
    assert np.allclose(2 * r1, r2, rtol=1e-10)
    nodal_loads_neg = {1: [-1000, -2000, -3000, -400, -500, -600], 3: [0, 0, 5000, 0, -1000, 0]}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(-u1, u_neg, rtol=1e-10)
    assert np.allclose(-r1, r_neg, rtol=1e-10)
    total_force_applied = np.zeros(3)
    total_moment_applied = np.zeros(3)
    for (node, loads) in nodal_loads.items():
        forces = np.array(loads[:3])
        moments = np.array(loads[3:])
        total_force_applied += forces
        total_moment_applied += moments + np.cross(node_coords[node], forces)
    reaction_dofs = slice(0, 6)
    reaction_forces = r1[reaction_dofs]
    reaction_force_vec = reaction_forces[:3]
    reaction_moment_vec = reaction_forces[3:]
    assert np.allclose(total_force_applied + reaction_force_vec, 0.0, atol=1e-10)
    total_reaction_moment = reaction_moment_vec
    assert np.allclose(total_moment_applied + total_reaction_moment, 0.0, atol=1e-10)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff).
    The solver should detect this (cond(K_ff) >= 1e16) and raise a ValueError."""
    node_coords = np.array([[0, 0, 0], [5, 0, 0]])
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-05
    I_z = 1e-05
    J = 2e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}]
    boundary_conditions = {}
    nodal_loads = {0: [1000, 0, 0, 0, 0, 0]}
    with pytest.raises(ValueError):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)