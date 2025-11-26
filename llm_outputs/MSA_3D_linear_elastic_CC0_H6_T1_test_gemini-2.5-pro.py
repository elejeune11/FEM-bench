def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L = 10.0
    E = 200000000000.0
    nu = 0.3
    rad = 0.05
    A = np.pi * rad ** 2
    I = np.pi * rad ** 4 / 4
    J = 2 * I
    P = 1000.0
    n_elem = 10
    n_nodes = n_elem + 1
    beam_axis_vec = np.array([1.0, 1.0, 1.0])
    beam_axis_unit = beam_axis_vec / np.linalg.norm(beam_axis_vec)
    node_coords = np.array([i * (L / n_elem) * beam_axis_unit for i in range(n_nodes)])
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    force_dir = np.array([1.0, -1.0, 0.0])
    force_dir_unit = force_dir / np.linalg.norm(force_dir)
    nodal_loads = {n_nodes - 1: np.concatenate((P * force_dir_unit, [0, 0, 0]))}
    delta_analytical_mag = P * L ** 3 / (3 * E * I)
    u_tip_analytical = delta_analytical_mag * force_dir_unit
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    u_tip_computed = u[(n_nodes - 1) * 6:(n_nodes - 1) * 6 + 3]
    assert np.linalg.norm(u_tip_computed) > 0
    assert np.allclose(u_tip_computed, u_tip_analytical, rtol=0.01)

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
    (Lx, Ly, H) = (2.0, 1.5, 1.0)
    node_coords = np.array([[0, 0, 0], [Lx, 0, 0], [Lx, Ly, 0], [0, Ly, 0], [0, 0, H], [Lx, 0, H], [Lx, Ly, H], [0, Ly, H]])
    element_nodes = [(0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (5, 6), (6, 7), (7, 4)]
    props = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    elements = [{'node_i': i, 'node_j': j, **props} for (i, j) in element_nodes]
    fixed_nodes = [0, 1, 2, 3]
    boundary_conditions = {i: [1, 1, 1, 1, 1, 1] for i in fixed_nodes}
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.all(u_zero == 0)
    assert np.all(r_zero == 0)
    load_node = 6
    load_vec = np.array([10000.0, -20000.0, 5000.0, 3000.0, 4000.0, -2000.0])
    nodal_loads_1 = {load_node: load_vec}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads_1)
    assert not np.all(u1 == 0)
    assert not np.all(r1 == 0)
    nodal_loads_2 = {load_node: 2 * load_vec}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_2)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    nodal_loads_3 = {load_node: -1 * load_vec}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_3)
    assert np.allclose(u3, -1 * u1)
    assert np.allclose(r3, -1 * r1)
    sum_applied_forces = np.zeros(3)
    sum_applied_moments_origin = np.zeros(3)
    for (node_idx, loads) in nodal_loads_1.items():
        sum_applied_forces += loads[0:3]
        sum_applied_moments_origin += loads[3:6]
        sum_applied_moments_origin += np.cross(node_coords[node_idx], loads[0:3])
    sum_reaction_forces = np.zeros(3)
    sum_reaction_moments_origin = np.zeros(3)
    for node_idx in fixed_nodes:
        reactions = r1[node_idx * 6:(node_idx + 1) * 6]
        sum_reaction_forces += reactions[0:3]
        sum_reaction_moments_origin += reactions[3:6]
        sum_reaction_moments_origin += np.cross(node_coords[node_idx], reactions[0:3])
    assert np.allclose(sum_applied_forces + sum_reaction_forces, 0, atol=1e-06)
    assert np.allclose(sum_applied_moments_origin + sum_reaction_moments_origin, 0, atol=1e-06)