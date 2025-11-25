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
    direction = np.array([1, 1, 1]) / np.sqrt(3)
    node_coords = np.array([i * L_elem * direction for i in range(n_elements + 1)])
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': 2 * I, 'local_z': np.array([0, 0, 1])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F = 1000.0
    force_direction = np.array([1, 1, -2]) / np.sqrt(6)
    nodal_loads = {n_elements: [F * force_direction[0], F * force_direction[1], F * force_direction[2], 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_disp = u[-6:-3]
    tip_disp_magnitude = np.linalg.norm(tip_disp)
    delta_analytical = F * L_total ** 3 / (3 * E * I)
    disp_direction = tip_disp / tip_disp_magnitude
    assert abs(np.dot(disp_direction, direction)) < 1e-10
    assert abs(tip_disp_magnitude - delta_analytical) / delta_analytical < 0.01

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0, 0, 0], [0, 0, 3], [4, 0, 3], [4, 0, 0], [2, 2, 3]])
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I = 1e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': 2 * I, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': 2 * I, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': 2 * I, 'local_z': None}, {'node_i': 1, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': 2 * I, 'local_z': None}, {'node_i': 2, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': 2 * I, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 3: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u1, 0, atol=1e-12)
    assert np.allclose(r1, 0, atol=1e-12)
    nodal_loads = {4: [1000, 500, -2000, 100, -50, 200]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.linalg.norm(u2) > 1e-06
    assert np.linalg.norm(r2) > 1e-06
    nodal_loads_double = {4: [2000, 1000, -4000, 200, -100, 400]}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u3, 2 * u2, rtol=1e-10)
    assert np.allclose(r3, 2 * r2, rtol=1e-10)
    nodal_loads_neg = {4: [-1000, -500, 2000, -100, 50, -200]}
    (u4, r4) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u4, -u2, rtol=1e-10)
    assert np.allclose(r4, -r2, rtol=1e-10)
    total_applied_force = np.array([1000, 500, -2000])
    total_applied_moment = np.array([100, -50, 200])
    reaction_forces = np.zeros(3)
    reaction_moments = np.zeros(3)
    for node_idx in boundary_conditions:
        dof_start = node_idx * 6
        reaction_forces += r2[dof_start:dof_start + 3]
        reaction_moments += r2[dof_start + 3:dof_start + 6]
        node_pos = node_coords[node_idx]
        reaction_moments += np.cross(node_pos, r2[dof_start:dof_start + 3])
    load_point = node_coords[4]
    total_applied_moment_about_origin = total_applied_moment + np.cross(load_point, total_applied_force)
    assert np.allclose(reaction_forces + total_applied_force, 0, atol=1e-08)
    assert np.allclose(reaction_moments + total_applied_moment_about_origin, 0, atol=1e-08)