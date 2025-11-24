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
    axis_dir = np.array([1, 1, 1])
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    node_coords = []
    for i in range(n_elements + 1):
        coord = i * L_element * axis_dir
        node_coords.append(coord)
    node_coords = np.array(node_coords)
    elements = []
    for i in range(n_elements):
        element = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}
        elements.append(element)
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    tip_node = n_elements
    F_magnitude = 1000.0
    arbitrary_vec = np.array([0, 0, 1])
    if np.allclose(axis_dir, arbitrary_vec):
        arbitrary_vec = np.array([1, 0, 0])
    transverse_dir = np.cross(axis_dir, arbitrary_vec)
    transverse_dir = transverse_dir / np.linalg.norm(transverse_dir)
    force_vector = F_magnitude * transverse_dir
    nodal_loads = {tip_node: [force_vector[0], force_vector[1], force_vector[2], 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    I_min = min(I_y, I_z)
    expected_deflection_magnitude = F_magnitude * L_total ** 3 / (3 * E * I_min)
    tip_dof_start = tip_node * 6
    tip_displacement = u[tip_dof_start:tip_dof_start + 3]
    computed_deflection_magnitude = np.linalg.norm(tip_displacement)
    tolerance = 0.05 * expected_deflection_magnitude
    assert abs(computed_deflection_magnitude - expected_deflection_magnitude) < tolerance, f'Tip deflection {computed_deflection_magnitude:.6f} differs from analytical {expected_deflection_magnitude:.6f}'

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0], [0, 5, 0], [5, 0, 3], [5, 5, 3], [0, 5, 3]])
    elements = []
    element_props = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.02, 'I_y': 1.67e-05, 'I_z': 1.67e-05, 'J': 3.33e-05, 'local_z': None}
    connections = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 3), (4, 1), (0, 4), (1, 5), (2, 6), (3, 6)]
    for (i, j) in connections:
        element = element_props.copy()
        element.update({'node_i': i, 'node_j': j})
        elements.append(element)
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0.0, atol=1e-10), 'Non-zero displacements with zero loads'
    assert np.allclose(r_zero, 0.0, atol=1e-10), 'Non-zero reactions with zero loads'
    nodal_loads_base = {1: [10000, 5000, -2000, 0, 0, 0], 2: [0, 0, -5000, 1000, -500, 200], 4: [-3000, 2000, -1000, 0, 300, -200]}
    (u_base, r_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads_base)
    assert not np.allclose(u_base, 0.0, atol=1e-10), 'Zero displacements with applied loads'
    fixed_node_dofs = slice(0, 6)
    assert not np.allclose(r_base[fixed_node_dofs], 0.0, atol=1e-10), 'Zero reactions with applied loads'
    nodal_loads_double = {}
    for (node, loads) in nodal_loads_base.items():
        nodal_loads_double[node] = [2 * load for load in loads]
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u_double, 2 * u_base, rtol=1e-08), 'Linearity violated for displacements'
    assert np.allclose(r_double, 2 * r_base, rtol=1e-08), 'Linearity violated for reactions'
    nodal_loads_negated = {}
    for (node, loads) in nodal_loads_base.items():
        nodal_loads_negated[node] = [-load for load in loads]
    (u_negated, r_negated) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negated)
    assert np.allclose(u_negated, -u_base, rtol=1e-08), 'Sign reversal violated for displacements'
    assert np.allclose(r_negated, -r_base, rtol=1e-08), 'Sign reversal violated for reactions'
    total_applied_force = np.zeros(3)
    total_applied_moment = np.zeros(3)
    for (node, loads) in nodal_loads_base.items():
        force = np.array(loads[0:3])
        moment = np.array(loads[3:6])
        position = node_coords[node]
        total_applied_force += force
        total_applied_moment += moment + np.cross(position, force)
    total_reaction_force = np.zeros(3)
    total_reaction_moment = np.zeros(3)
    for (node, bc) in boundary_conditions.items():
        if any(bc):
            node_dofs = slice(node * 6, node * 6 + 6)
            reaction = r_base[node_dofs]
            reaction_force = reaction[0:3]
            reaction_moment = reaction[3:6]
            position = node_coords[node]
            total_reaction_force += reaction_force
            total_reaction_moment += reaction_moment + np.cross(position, reaction_force)
    assert np.allclose(total_applied_force + total_reaction_force, 0.0, atol=1e-08), 'Force equilibrium not satisfied'
    assert np.allclose(total_applied_moment + total_reaction_moment, 0.0, atol=1e-08), 'Moment equilibrium not satisfied'