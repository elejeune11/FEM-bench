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
    axis_dir = np.array([1.0, 1.0, 1.0])
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    n_nodes = n_elements + 1
    node_coords = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        node_coords[i] = i * L_total / n_elements * axis_dir
    local_z = np.array([1.0, -1.0, 0.0])
    local_z = local_z / np.linalg.norm(local_z)
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    local_y = np.cross(local_z, axis_dir)
    local_y = local_y / np.linalg.norm(local_y)
    P = 1000.0
    F_tip = P * local_z
    nodal_loads = {n_nodes - 1: [F_tip[0], F_tip[1], F_tip[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    delta_analytical = P * L_total ** 3 / (3 * E * I_z)
    tip_node = n_nodes - 1
    tip_disp = u[tip_node * 6:tip_node * 6 + 3]
    tip_deflection_in_load_dir = np.dot(tip_disp, local_z)
    rel_error = abs(tip_deflection_in_load_dir - delta_analytical) / delta_analytical
    assert rel_error < 0.02, f'Tip deflection error {rel_error * 100:.2f}% exceeds 2% tolerance'
    fixed_disp = u[0:6]
    assert np.allclose(fixed_disp, 0.0, atol=1e-12), 'Fixed DOFs should have zero displacement'
    reactions_at_fixed = r[0:6]
    assert not np.allclose(reactions_at_fixed, 0.0), 'Reactions should be nonzero at fixed node'

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
    node_coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.5, 2.5, 0.0], [1.5, 1.0, 3.0]])
    n_nodes = len(node_coords)
    E = 210000000000.0
    nu = 0.3
    A = 0.005
    I_y = 5e-05
    I_z = 5e-05
    J = 0.0001
    elements = [{'node_i': 0, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 1: [1, 1, 1, 1, 1, 1], 2: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0.0, atol=1e-12), 'Zero loads should produce zero displacements'
    assert np.allclose(r_zero, 0.0, atol=1e-12), 'Zero loads should produce zero reactions'
    F_applied = [1000.0, -500.0, 2000.0, 100.0, -200.0, 150.0]
    nodal_loads_base = {3: F_applied}
    (u_base, r_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads_base)
    free_node_disp = u_base[3 * 6:4 * 6]
    assert not np.allclose(free_node_disp, 0.0), 'Free node should have nonzero displacement under load'
    reactions_fixed = np.concatenate([r_base[0:6], r_base[6:12], r_base[12:18]])
    assert not np.allclose(reactions_fixed, 0.0), 'Fixed nodes should have nonzero reactions'
    F_doubled = [2 * f for f in F_applied]
    nodal_loads_doubled = {3: F_doubled}
    (u_doubled, r_doubled) = fcn(node_coords, elements, boundary_conditions, nodal_loads_doubled)
    assert np.allclose(u_doubled, 2 * u_base, rtol=1e-10), 'Doubling loads should double displacements'
    assert np.allclose(r_doubled, 2 * r_base, rtol=1e-10), 'Doubling loads should double reactions'
    F_negated = [-f for f in F_applied]
    nodal_loads_negated = {3: F_negated}
    (u_negated, r_negated) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negated)
    assert np.allclose(u_negated, -u_base, rtol=1e-10), 'Negating loads should negate displacements'
    assert np.allclose(r_negated, -r_base, rtol=1e-10), 'Negating loads should negate reactions'
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    for (node_idx, loads) in nodal_loads_base.items():
        total_force += np.array(loads[0:3])
        total_moment += np.array(loads[3:6])
        pos = node_coords[node_idx]
        total_moment += np.cross(pos, np.array(loads[0:3]))
    for node_idx in range(n_nodes):
        reaction_force = r_base[node_idx * 6:node_idx * 6 + 3]
        reaction_moment = r_base[node_idx * 6 + 3:node_idx * 6 + 6]
        total_force += reaction_force
        total_moment += reaction_moment
        pos = node_coords[node_idx]
        total_moment += np.cross(pos, reaction_force)
    assert np.allclose(total_force, 0.0, atol=1e-06), 'Global force equilibrium not satisfied'
    assert np.allclose(total_moment, 0.0, atol=1e-06), 'Global moment equilibrium not satisfied'