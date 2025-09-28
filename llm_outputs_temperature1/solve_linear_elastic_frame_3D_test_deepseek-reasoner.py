def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.33e-06
    I_z = 8.33e-06
    J = 1.67e-05
    n_elements = 10
    n_nodes = n_elements + 1
    node_coords = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        factor = i * L / n_elements
        node_coords[i] = [factor, factor, factor]
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_magnitude = 1000.0
    perpendicular_force = F_magnitude * np.array([1, -1, 0]) / np.sqrt(2)
    nodal_loads = {n_nodes - 1: [perpendicular_force[0], perpendicular_force[1], perpendicular_force[2], 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    analytical_deflection = F_magnitude * L ** 3 / (3 * E * I_z)
    tip_disp = u[(n_nodes - 1) * 6:(n_nodes - 1) * 6 + 3]
    tip_disp_magnitude = np.linalg.norm(tip_disp)
    assert tip_disp_magnitude == pytest.approx(analytical_deflection, rel=0.05)
    total_applied_force = np.array([perpendicular_force[0], perpendicular_force[1], perpendicular_force[2], 0, 0, 0])
    total_reaction = np.sum(r.reshape(-1, 6), axis=0)
    assert np.allclose(total_reaction[:3] + total_applied_force[:3], 0.0, atol=1e-06)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0], [5, 5, 5]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0.0, atol=1e-10)
    assert np.allclose(r_zero, 0.0, atol=1e-10)
    nodal_loads_base = {1: [1000, 0, 0, 0, 0, 0], 2: [0, 2000, 0, 500, 0, 0], 3: [0, 0, 3000, 0, 600, 0]}
    (u_base, r_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads_base)
    assert not np.allclose(u_base, 0.0, atol=1e-10)
    assert not np.allclose(r_base, 0.0, atol=1e-10)
    nodal_loads_double = {1: [2000, 0, 0, 0, 0, 0], 2: [0, 4000, 0, 1000, 0, 0], 3: [0, 0, 6000, 0, 1200, 0]}
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u_double, 2.0 * u_base, rtol=1e-06)
    assert np.allclose(r_double, 2.0 * r_base, rtol=1e-06)
    nodal_loads_negated = {1: [-1000, 0, 0, 0, 0, 0], 2: [0, -2000, 0, -500, 0, 0], 3: [0, 0, -3000, 0, -600, 0]}
    (u_negated, r_negated) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negated)
    assert np.allclose(u_negated, -u_base, rtol=1e-06)
    assert np.allclose(r_negated, -r_base, rtol=1e-06)
    total_applied_force = np.zeros(6)
    for (node, loads) in nodal_loads_base.items():
        total_applied_force += np.array(loads)
    total_reaction = np.sum(r_base.reshape(-1, 6), axis=0)
    assert np.allclose(total_applied_force + total_reaction, 0.0, atol=1e-06)
    total_moment = np.zeros(3)
    for (node, loads) in nodal_loads_base.items():
        force = np.array(loads[:3])
        moment = np.array(loads[3:6])
        position = node_coords[node]
        total_moment += moment + np.cross(position, force)
    total_reaction_moment = np.zeros(3)
    for i in range(len(node_coords)):
        reaction_force = r_base[i * 6:i * 6 + 3]
        reaction_moment = r_base[i * 6 + 3:i * 6 + 6]
        position = node_coords[i]
        total_reaction_moment += reaction_moment + np.cross(position, reaction_force)
    assert np.allclose(total_moment + total_reaction_moment, 0.0, atol=1e-06)