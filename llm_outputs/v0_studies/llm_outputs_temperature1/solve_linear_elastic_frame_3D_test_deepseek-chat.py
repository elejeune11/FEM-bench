def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L_total = 10.0
    n_elements = 10
    L_element = L_total / n_elements
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
        pos = i * L_element * beam_axis
        node_coords.append(pos)
    node_coords = np.array(node_coords)
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    arbitrary_vec = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.cross(beam_axis, arbitrary_vec), 0):
        arbitrary_vec = np.array([0.0, 1.0, 0.0])
    force_dir = np.cross(beam_axis, arbitrary_vec)
    force_dir = force_dir / np.linalg.norm(force_dir)
    F_magnitude = 1000.0
    force_vector = F_magnitude * force_dir
    nodal_loads = {n_elements: [force_vector[0], force_vector[1], force_vector[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    I_effective = I_z
    analytical_deflection = F_magnitude * L_total ** 3 / (3 * E * I_effective)
    tip_disp = u[6 * n_elements:6 * n_elements + 3]
    numerical_deflection = np.linalg.norm(tip_disp)
    assert abs(numerical_deflection - analytical_deflection) / analytical_deflection < 0.02

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0, 0, 0], [5, 0, 0], [5, 4, 0], [0, 4, 0], [5, 0, 3], [0, 4, 3]])
    elements = []
    material_props = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.02, 'I_y': 1.67e-05, 'I_z': 1.67e-05, 'J': 3.33e-05}
    connections = [(0, 1), (1, 2), (2, 3), (1, 4), (3, 5), (4, 5), (2, 5)]
    for (i, j) in connections:
        elements.append({'node_i': i, 'node_j': j, 'local_z': None, **material_props})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0.0), 'Zero loads should produce zero displacements'
    fixed_dofs = [0, 1, 2, 3, 4, 5]
    assert np.allclose(r_zero[fixed_dofs], 0.0), 'Zero loads should produce zero reactions'
    nodal_loads_base = {2: [10.0, -5.0, 8.0, 2.0, -3.0, 1.0], 4: [0.0, 0.0, -15.0, 0.0, 4.0, 0.0], 5: [7.0, 3.0, 0.0, -2.0, 0.0, 1.5]}
    (u_base, r_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads_base)
    free_dofs = list(set(range(6 * len(node_coords))) - set(fixed_dofs))
    assert not np.allclose(u_base[free_dofs], 0.0), 'Nonzero loads should produce nonzero displacements'
    assert not np.allclose(r_base[fixed_dofs], 0.0), 'Nonzero loads should produce nonzero reactions'
    nodal_loads_double = {}
    for (node, loads) in nodal_loads_base.items():
        nodal_loads_double[node] = [2.0 * load for load in loads]
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u_double, 2.0 * u_base, rtol=1e-10), 'Doubling loads should double displacements'
    assert np.allclose(r_double, 2.0 * r_base, rtol=1e-10), 'Doubling loads should double reactions'
    nodal_loads_negated = {}
    for (node, loads) in nodal_loads_base.items():
        nodal_loads_negated[node] = [-load for load in loads]
    (u_negated, r_negated) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negated)
    assert np.allclose(u_negated, -u_base, rtol=1e-10), 'Negating loads should negate displacements'
    assert np.allclose(r_negated, -r_base, rtol=1e-10), 'Negating loads should negate reactions'
    total_force_applied = np.zeros(3)
    total_moment_applied = np.zeros(3)
    for (node, loads) in nodal_loads_base.items():
        force = np.array(loads[:3])
        moment = np.array(loads[3:])
        position = node_coords[node]
        total_force_applied += force
        total_moment_applied += moment + np.cross(position, force)
    reaction_force = np.array(r_base[0:3])
    reaction_moment = np.array(r_base[3:6])
    force_balance = total_force_applied + reaction_force
    assert np.allclose(force_balance, 0.0, atol=1e-10), 'Force equilibrium not satisfied'
    moment_balance = total_moment_applied + reaction_moment
    assert np.allclose(moment_balance, 0.0, atol=1e-10), 'Moment equilibrium not satisfied'