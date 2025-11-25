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
    I_y = 8.33e-06
    I_z = 8.33e-06
    J = 1.67e-05
    beam_axis = np.array([1.0, 1.0, 1.0])
    beam_axis = beam_axis / np.linalg.norm(beam_axis)
    node_coords = []
    for i in range(n_elements + 1):
        pos = i * element_length * beam_axis
        node_coords.append(pos)
    node_coords = np.array(node_coords)
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    tip_node = n_elements
    F_magnitude = 1000.0
    if abs(beam_axis[0]) > 0.5:
        arbitrary_vec = np.array([0.0, 1.0, 0.0])
    else:
        arbitrary_vec = np.array([1.0, 0.0, 0.0])
    transverse_dir = np.cross(beam_axis, arbitrary_vec)
    transverse_dir = transverse_dir / np.linalg.norm(transverse_dir)
    F_vector = F_magnitude * transverse_dir
    nodal_loads = {tip_node: [F_vector[0], F_vector[1], F_vector[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    I_effective = (I_y + I_z) / 2
    analytical_deflection_magnitude = F_magnitude * L ** 3 / (3 * E * I_effective)
    tip_dofs = slice(6 * tip_node, 6 * tip_node + 3)
    tip_displacement = u[tip_dofs]
    numerical_deflection_magnitude = np.linalg.norm(tip_displacement)
    assert abs(numerical_deflection_magnitude - analytical_deflection_magnitude) / analytical_deflection_magnitude < 0.02

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 0.0], [0.0, 5.0, 0.0], [2.5, 2.5, 3.0], [0.0, 0.0, 6.0]])
    E = 200000000000.0
    nu = 0.3
    A = 0.02
    I_y = 1.67e-05
    I_z = 1.67e-05
    J = 3.33e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 3, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 0, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 0, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 2, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 0, 'node_j': 5, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 3: [1, 1, 1, 1, 1, 1]}
    n_nodes = len(node_coords)
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0.0, atol=1e-12)
    assert np.allclose(r_zero, 0.0, atol=1e-12)
    nodal_loads_base = {1: [10000.0, 5000.0, -2000.0, 0.0, 0.0, 0.0], 2: [0.0, 0.0, -5000.0, 1000.0, 2000.0, -500.0], 4: [3000.0, -2000.0, -8000.0, 0.0, 0.0, 0.0], 5: [0.0, 0.0, -10000.0, 0.0, 0.0, 0.0]}
    (u_base, r_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads_base)
    assert not np.allclose(u_base, 0.0, atol=1e-12)
    fixed_dofs_node0 = slice(0, 6)
    fixed_dofs_node3 = slice(18, 24)
    assert not np.allclose(r_base[fixed_dofs_node0], 0.0, atol=1e-06)
    assert not np.allclose(r_base[fixed_dofs_node3], 0.0, atol=1e-06)
    nodal_loads_double = {}
    for (node, loads) in nodal_loads_base.items():
        nodal_loads_double[node] = [2.0 * load for load in loads]
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u_double, 2.0 * u_base, rtol=1e-10)
    assert np.allclose(r_double, 2.0 * r_base, rtol=1e-10)
    nodal_loads_negated = {}
    for (node, loads) in nodal_loads_base.items():
        nodal_loads_negated[node] = [-load for load in loads]
    (u_negated, r_negated) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negated)
    assert np.allclose(u_negated, -u_base, rtol=1e-10)
    assert np.allclose(r_negated, -r_base, rtol=1e-10)
    total_applied_force = np.zeros(3)
    total_applied_moment = np.zeros(3)
    for (node, loads) in nodal_loads_base.items():
        force = np.array(loads[0:3])
        moment = np.array(loads[3:6])
        total_applied_force += force
        total_applied_moment += moment + np.cross(node_coords[node], force)
    total_reaction_force = np.zeros(3)
    total_reaction_moment = np.zeros(3)
    for fixed_node in boundary_conditions.keys():
        reaction_start = 6 * fixed_node
        reaction_force = r_base[reaction_start:reaction_start + 3]
        reaction_moment = r_base[reaction_start + 3:reaction_start + 6]
        total_reaction_force += reaction_force
        total_reaction_moment += reaction_moment + np.cross(node_coords[fixed_node], reaction_force)
    assert np.allclose(total_applied_force + total_reaction_force, 0.0, atol=1e-08)
    assert np.allclose(total_applied_moment + total_reaction_moment, 0.0, atol=1e-08)