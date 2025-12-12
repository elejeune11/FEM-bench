def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L_total = 10.0
    n_elements = 10
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 0.0001
    I_z = 0.0001
    J = 0.0002
    axis_dir = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    n_nodes = n_elements + 1
    node_coords = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        node_coords[i] = i * L_total / n_elements * axis_dir
    elements = []
    local_z = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    P = 1000.0
    force_dir = local_z
    nodal_loads = {n_nodes - 1: [P * force_dir[0], P * force_dir[1], P * force_dir[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    delta_analytical = P * L_total ** 3 / (3 * E * I_y)
    tip_node = n_nodes - 1
    tip_disp = u[tip_node * 6:tip_node * 6 + 3]
    tip_deflection_in_load_dir = np.dot(tip_disp, force_dir)
    relative_error = abs(tip_deflection_in_load_dir - delta_analytical) / delta_analytical
    assert relative_error < 0.02, f'Tip deflection {tip_deflection_in_load_dir} differs from analytical {delta_analytical} by {relative_error * 100:.2f}%'
    fixed_disp = u[0:6]
    assert np.allclose(fixed_disp, 0.0, atol=1e-12), 'Fixed node should have zero displacement'
    fixed_reactions = r[0:6]
    assert not np.allclose(fixed_reactions, 0.0, atol=1e-06), 'Fixed node should have nonzero reactions'

def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 4.0], [5.0, 0.0, 4.0]])
    E = 210000000000.0
    nu = 0.3
    A = 0.02
    I_y = 0.0005
    I_z = 0.0005
    J = 0.001
    elements = [{'node_i': 0, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 1: [1, 1, 1, 1, 1, 1]}
    n_nodes = 4
    n_dofs = 6 * n_nodes
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0.0, atol=1e-12), 'Zero loads should produce zero displacements'
    assert np.allclose(r_zero, 0.0, atol=1e-12), 'Zero loads should produce zero reactions'
    nodal_loads = {2: [1000.0, 500.0, -200.0, 50.0, -30.0, 20.0], 3: [-500.0, 800.0, 300.0, -40.0, 60.0, -10.0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    free_disp = np.concatenate([u1[12:18], u1[18:24]])
    assert not np.allclose(free_disp, 0.0, atol=1e-12), 'Free nodes should have nonzero displacements under load'
    fixed_reactions = np.concatenate([r1[0:6], r1[6:12]])
    assert not np.allclose(fixed_reactions, 0.0, atol=1e-12), 'Fixed nodes should have nonzero reactions under load'
    assert np.allclose(u1[0:6], 0.0, atol=1e-12), 'Fixed node 0 should have zero displacement'
    assert np.allclose(u1[6:12], 0.0, atol=1e-12), 'Fixed node 1 should have zero displacement'
    nodal_loads_double = {2: [2 * x for x in nodal_loads[2]], 3: [2 * x for x in nodal_loads[3]]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u2, 2 * u1, rtol=1e-10), 'Doubling loads should double displacements'
    assert np.allclose(r2, 2 * r1, rtol=1e-10), 'Doubling loads should double reactions'
    nodal_loads_neg = {2: [-x for x in nodal_loads[2]], 3: [-x for x in nodal_loads[3]]}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u_neg, -u1, rtol=1e-10), 'Negating loads should negate displacements'
    assert np.allclose(r_neg, -r1, rtol=1e-10), 'Negating loads should negate reactions'
    total_applied_force = np.zeros(3)
    total_applied_moment = np.zeros(3)
    for (node_idx, loads) in nodal_loads.items():
        force = np.array(loads[0:3])
        moment = np.array(loads[3:6])
        pos = node_coords[node_idx]
        total_applied_force += force
        total_applied_moment += moment + np.cross(pos, force)
    total_reaction_force = np.zeros(3)
    total_reaction_moment = np.zeros(3)
    for node_idx in [0, 1]:
        reaction_force = r1[node_idx * 6:node_idx * 6 + 3]
        reaction_moment = r1[node_idx * 6 + 3:node_idx * 6 + 6]
        pos = node_coords[node_idx]
        total_reaction_force += reaction_force
        total_reaction_moment += reaction_moment + np.cross(pos, reaction_force)
    net_force = total_applied_force + total_reaction_force
    net_moment = total_applied_moment + total_reaction_moment
    assert np.allclose(net_force, 0.0, atol=1e-06), f'Force equilibrium not satisfied: {net_force}'
    assert np.allclose(net_moment, 0.0, atol=1e-06), f'Moment equilibrium not satisfied: {net_moment}'