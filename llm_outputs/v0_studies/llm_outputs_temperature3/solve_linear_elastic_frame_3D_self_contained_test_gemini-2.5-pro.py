def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    E = 200000000000.0
    nu = 0.3
    radius = 0.1
    A = np.pi * radius ** 2
    I = np.pi * radius ** 4 / 4
    J = np.pi * radius ** 4 / 2
    num_elements = 10
    num_nodes = num_elements + 1
    beam_axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    node_coords = np.array([i * (L / num_elements) * beam_axis for i in range(num_nodes)])
    elements = []
    local_z_ref = np.array([0.0, 0.0, 1.0])
    for i in range(num_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z_ref})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    P_mag = 1000.0
    force_dir = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    force_vec = P_mag * force_dir
    nodal_loads = {num_nodes - 1: list(force_vec) + [0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    analytical_deflection = P_mag * L ** 3 / (3 * E * I)
    tip_node_idx = num_nodes - 1
    tip_disp_vec = u[tip_node_idx * 6:tip_node_idx * 6 + 3]
    computed_deflection_mag = np.linalg.norm(tip_disp_vec)
    assert np.isclose(computed_deflection_mag, analytical_deflection, rtol=0.02)
    computed_dir = tip_disp_vec / computed_deflection_mag
    assert np.allclose(computed_dir, force_dir, atol=0.001)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
Suggested Test stages:
1. Zero loads -> All displacements and reactions should be zero.
2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
3. Double the loads -> Displacements and reactions should double (linearity check).
4. Negate the original loads -> Displacements and reactions should flip sign.
5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    num_nodes = len(node_coords)
    props = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    elements = [{'node_i': 0, 'node_j': 1, **props, 'local_z': [0, 0, 1]}, {'node_i': 1, 'node_j': 2, **props, 'local_z': [0, 0, 1]}, {'node_i': 2, 'node_j': 3, **props, 'local_z': [1, 0, 0]}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, {})
    assert np.all(u_zero == 0)
    assert np.all(r_zero == 0)
    nodal_loads_base = {3: [1000.0, 2000.0, -1500.0, 500.0, -300.0, 100.0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads_base)
    assert not np.all(u1 == 0)
    assert not np.all(r1 == 0)
    nodal_loads_double = {k: [2 * val for val in v] for (k, v) in nodal_loads_base.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    nodal_loads_neg = {k: [-val for val in v] for (k, v) in nodal_loads_base.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u3, -u1)
    assert np.allclose(r3, -r1)
    total_applied_force = np.sum([v[:3] for v in nodal_loads_base.values()], axis=0)
    total_reaction_force = np.sum(r1.reshape(num_nodes, 6)[:, :3], axis=0)
    assert np.allclose(total_applied_force + total_reaction_force, 0, atol=1e-09)
    total_applied_moment = np.sum([v[3:] for v in nodal_loads_base.values()], axis=0)
    total_reaction_moment = np.sum(r1.reshape(num_nodes, 6)[:, 3:], axis=0)
    moment_from_applied_forces = np.zeros(3)
    for (node_idx, load) in nodal_loads_base.items():
        pos = node_coords[node_idx]
        force = load[:3]
        moment_from_applied_forces += np.cross(pos, force)
    moment_from_reaction_forces = np.zeros(3)
    r1_reshaped = r1.reshape(num_nodes, 6)
    for i in range(num_nodes):
        pos = node_coords[i]
        reaction_force = r1_reshaped[i, :3]
        moment_from_reaction_forces += np.cross(pos, reaction_force)
    total_moment = total_applied_moment + moment_from_applied_forces + total_reaction_moment + moment_from_reaction_forces
    assert np.allclose(total_moment, 0, atol=1e-09)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
when the structure is improperly constrained, leading to an
ill-conditioned free-free stiffness matrix (K_ff).
The solver should detect this (cond(K_ff) >= 1e16) and raise a ValueError."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': [0, 0, 1]}]
    boundary_conditions = {}
    nodal_loads = {1: [0, 1000, 0, 0, 0, 0]}
    with pytest.raises(ValueError):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)