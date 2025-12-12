def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L_total = 10.0
    n_elements = 10
    L_elem = L_total / n_elements
    node_coords = np.array([[i * L_elem / np.sqrt(3), i * L_elem / np.sqrt(3), i * L_elem / np.sqrt(3)] for i in range(n_elements + 1)], dtype=float)
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.333e-05
    I_z = 8.333e-05
    J = 0.0001667
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_mag = 1000.0
    load_direction = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    nodal_loads = {n_elements: [F_mag * load_direction[0], F_mag * load_direction[1], F_mag * load_direction[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    I_eff = I_y
    delta_analytical = F_mag * L_total ** 3 / (3.0 * E * I_eff)
    tip_node_idx = n_elements
    tip_disp = u[6 * tip_node_idx:6 * tip_node_idx + 3]
    tip_disp_magnitude = np.linalg.norm(tip_disp)
    assert np.isclose(tip_disp_magnitude, delta_analytical, rtol=0.05), f'Tip deflection {tip_disp_magnitude} does not match analytical {delta_analytical}'
    fixed_disp = u[0:6]
    assert np.allclose(fixed_disp, 0.0, atol=1e-10), 'Fixed node should have zero displacement'
    fixed_reactions = r[0:6]
    applied_load = np.array(nodal_loads[n_elements])
    assert np.allclose(fixed_reactions[0:3], -applied_load[0:3], atol=1e-06), 'Reactions do not balance applied forces'

def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.333e-05
    I_z = 8.333e-05
    J = 0.0001667
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0.0, atol=1e-12), 'Stage 1: Zero loads should result in zero displacements'
    assert np.allclose(r_zero, 0.0, atol=1e-12), 'Stage 1: Zero loads should result in zero reactions'
    nodal_loads_1 = {3: [100.0, 50.0, 75.0, 10.0, 5.0, 2.0]}
    (u_1, r_1) = fcn(node_coords, elements, boundary_conditions, nodal_loads_1)
    free_disp_1 = u_1[18:24]
    assert not np.allclose(free_disp_1, 0.0, atol=1e-10), 'Stage 2: Free node should have nonzero displacements under load'
    fixed_reactions_1 = r_1[0:6]
    assert not np.allclose(fixed_reactions_1, 0.0, atol=1e-10), 'Stage 2: Fixed node should have nonzero reactions'
    nodal_loads_2 = {3: [200.0, 100.0, 150.0, 20.0, 10.0, 4.0]}
    (u_2, r_2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_2)
    assert np.allclose(u_2, 2.0 * u_1, rtol=1e-10), 'Stage 3: Doubling loads should double displacements (linearity)'
    assert np.allclose(r_2, 2.0 * r_1, rtol=1e-10), 'Stage 3: Doubling loads should double reactions (linearity)'
    nodal_loads_neg = {3: [-100.0, -50.0, -75.0, -10.0, -5.0, -2.0]}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u_neg, -u_1, rtol=1e-10), 'Stage 4: Negating loads should negate displacements'
    assert np.allclose(r_neg, -r_1, rtol=1e-10), 'Stage 4: Negating loads should negate reactions'
    total_force = np.array(nodal_loads_1[3][0:3]) + r_1[0:3]
    total_moment = np.array(nodal_loads_1[3][3:6]) + r_1[3:6]
    assert np.allclose(total_force, 0.0, atol=1e-06), 'Stage 5: Total forces should satisfy equilibrium'
    assert np.allclose(total_moment, 0.0, atol=1e-06), 'Stage 5: Total moments should satisfy equilibrium'