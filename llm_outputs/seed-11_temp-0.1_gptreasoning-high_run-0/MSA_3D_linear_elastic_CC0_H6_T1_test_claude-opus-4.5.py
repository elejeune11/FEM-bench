def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L_total = 10.0
    n_elements = 10
    L_elem = L_total / n_elements
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
        node_coords[i] = i * L_elem * axis_dir
    temp_vec = np.array([1.0, 0.0, 0.0])
    local_z = np.cross(axis_dir, temp_vec)
    local_z = local_z / np.linalg.norm(local_z)
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    P = 1000.0
    force_dir = local_z
    nodal_loads = {n_nodes - 1: [P * force_dir[0], P * force_dir[1], P * force_dir[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    delta_analytical = P * L_total ** 3 / (3 * E * I_z)
    tip_node = n_nodes - 1
    tip_disp = u[tip_node * 6:tip_node * 6 + 3]
    tip_deflection = np.dot(tip_disp, force_dir)
    rel_error = abs(tip_deflection - delta_analytical) / delta_analytical
    assert rel_error < 0.02, f'Tip deflection {tip_deflection} differs from analytical {delta_analytical} by {rel_error * 100:.2f}%'
    fixed_disp = u[0:6]
    assert np.allclose(fixed_disp, 0.0, atol=1e-12), 'Fixed DOFs should have zero displacement'
    fixed_reactions = r[0:6]
    assert not np.allclose(fixed_reactions, 0.0), 'Reactions at fixed node should be nonzero'

def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 3.0], [4.0, 0.0, 3.0], [2.0, 2.0, 4.0]])
    n_nodes = len(node_coords)
    E = 210000000000.0
    nu = 0.3
    A = 0.005
    I_y = 5e-05
    I_z = 5e-05
    J = 0.0001
    elements = [{'node_i': 0, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 2, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 3, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 1: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0.0, atol=1e-12), 'Zero loads should produce zero displacements'
    assert np.allclose(r_zero, 0.0, atol=1e-12), 'Zero loads should produce zero reactions'
    nodal_loads_base = {2: [1000.0, -500.0, 200.0, 50.0, -30.0, 20.0], 3: [-800.0, 600.0, -300.0, -40.0, 25.0, -15.0], 4: [500.0, 400.0, 1000.0, 100.0, -50.0, 75.0]}
    (u_base, r_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads_base)
    free_node_indices = [2, 3, 4]
    for node_idx in free_node_indices:
        node_disp = u_base[node_idx * 6:(node_idx + 1) * 6]
        assert not np.allclose(node_disp, 0.0), f'Displacements at free node {node_idx} should be nonzero'
    fixed_node_indices = [0, 1]
    for node_idx in fixed_node_indices:
        node_react = r_base[node_idx * 6:(node_idx + 1) * 6]
        assert not np.allclose(node_react, 0.0), f'Reactions at fixed node {node_idx} should be nonzero'
    for node_idx in fixed_node_indices:
        node_disp = u_base[node_idx * 6:(node_idx + 1) * 6]
        assert np.allclose(node_disp, 0.0, atol=1e-12), f'Fixed node {node_idx} should have zero displacement'
    nodal_loads_double = {node: [2.0 * val for val in loads] for (node, loads) in nodal_loads_base.items()}
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u_double, 2.0 * u_base, rtol=1e-10), 'Doubling loads should double displacements'
    assert np.allclose(r_double, 2.0 * r_base, rtol=1e-10), 'Doubling loads should double reactions'
    nodal_loads_neg = {node: [-val for val in loads] for (node, loads) in nodal_loads_base.items()}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u_neg, -u_base, rtol=1e-10), 'Negating loads should negate displacements'
    assert np.allclose(r_neg, -r_base, rtol=1e-10), 'Negating loads should negate reactions'
    total_applied_force = np.zeros(3)
    total_applied_moment = np.zeros(3)
    for (node_idx, loads) in nodal_loads_base.items():
        total_applied_force += np.array(loads[0:3])
        total_applied_moment += np.array(loads[3:6])
        node_pos = node_coords[node_idx]
        total_applied_moment += np.cross(node_pos, np.array(loads[0:3]))
    total_reaction_force = np.zeros(3)
    total_reaction_moment = np.zeros(3)
    for node_idx in range(n_nodes):
        react = r_base[node_idx * 6:(node_idx + 1) * 6]
        total_reaction_force += react[0:3]
        total_reaction_moment += react[3:6]
        node_pos = node_coords[node_idx]
        total_reaction_moment += np.cross(node_pos, react[0:3])
    net_force = total_applied_force + total_reaction_force
    assert np.allclose(net_force, 0.0, atol=1e-06), f'Force equilibrium not satisfied: net force = {net_force}'
    net_moment = total_applied_moment + total_reaction_moment
    assert np.allclose(net_moment, 0.0, atol=1e-06), f'Moment equilibrium not satisfied: net moment = {net_moment}'