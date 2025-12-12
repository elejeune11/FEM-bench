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
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_mag = 1000.0
    F_dir = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    nodal_loads = {n_elements: [F_mag * F_dir[0], F_mag * F_dir[1], F_mag * F_dir[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    L_eff = L_total
    I_eff = I_y
    delta_analytical = F_mag * L_eff ** 3 / (3.0 * E * I_eff)
    tip_disp = u[n_elements * 6:n_elements * 6 + 3]
    tip_disp_mag = np.linalg.norm(tip_disp)
    assert abs(tip_disp_mag - delta_analytical) / delta_analytical < 0.05, f'Tip deflection {tip_disp_mag} does not match analytical solution {delta_analytical}'
    fixed_disp = u[0:6]
    assert np.allclose(fixed_disp, 0.0, atol=1e-12), 'Fixed node should have zero displacement'

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
    nodal_loads_1 = {3: [100.0, 50.0, 25.0, 10.0, 5.0, 2.0]}
    (u_1, r_1) = fcn(node_coords, elements, boundary_conditions, nodal_loads_1)
    assert not np.allclose(u_1, 0.0, atol=1e-06), 'Stage 2: Applied loads should produce nonzero displacements'
    assert not np.allclose(r_1, 0.0, atol=1e-06), 'Stage 2: Applied loads should produce nonzero reactions'
    nodal_loads_2 = {3: [200.0, 100.0, 50.0, 20.0, 10.0, 4.0]}
    (u_2, r_2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_2)
    assert np.allclose(u_2, 2.0 * u_1, rtol=1e-10), 'Stage 3: Doubling loads should double displacements (linearity)'
    assert np.allclose(r_2, 2.0 * r_1, rtol=1e-10), 'Stage 3: Doubling loads should double reactions (linearity)'
    nodal_loads_neg = {3: [-100.0, -50.0, -25.0, -10.0, -5.0, -2.0]}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u_neg, -u_1, rtol=1e-10), 'Stage 4: Negating loads should negate displacements'
    assert np.allclose(r_neg, -r_1, rtol=1e-10), 'Stage 4: Negating loads should negate reactions'
    total_force = np.array([100.0, 50.0, 25.0, 0.0, 0.0, 0.0])
    reaction_force = r_1[0:6]
    applied_forces = total_force[0:3]
    applied_moments = total_force[3:6]
    reaction_forces = reaction_force[0:3]
    reaction_moments = reaction_force[3:6]
    force_equilibrium = applied_forces + reaction_forces
    assert np.allclose(force_equilibrium, 0.0, atol=1e-06), f'Stage 5: Force equilibrium violated. Sum = {force_equilibrium}'
    r_vec = node_coords[3] - node_coords[0]
    moment_from_force = np.cross(r_vec, applied_forces)
    moment_equilibrium = applied_moments + reaction_moments + moment_from_force
    assert np.allclose(moment_equilibrium, 0.0, atol=0.0001), f'Stage 5: Moment equilibrium violated. Sum = {moment_equilibrium}'