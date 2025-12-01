def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L = 10.0
    num_elements = 10
    num_nodes = num_elements + 1
    E = 200000000000.0
    nu = 0.3
    r = 0.1
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = np.pi * r ** 4 / 2
    axis = np.array([1, 1, 1], dtype=float)
    axis_normalized = axis / np.linalg.norm(axis)
    node_coords = np.zeros((num_nodes, 3))
    for i in range(num_nodes):
        distance = i / num_elements * L
        node_coords[i] = distance * axis_normalized
    elements = []
    for i in range(num_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    load_dir = np.array([1, -1, 0], dtype=float)
    load_dir /= np.linalg.norm(load_dir)
    P = 1000.0
    F_vec = P * load_dir
    nodal_loads = {num_nodes - 1: [F_vec[0], F_vec[1], F_vec[2], 0, 0, 0]}
    (u, r_reac) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_dofs = u[-6:]
    tip_translation = tip_dofs[:3]
    tip_deflection = np.linalg.norm(tip_translation)
    analytical_deflection = P * L ** 3 / (3 * E * I)
    assert np.isclose(tip_deflection, analytical_deflection, rtol=0.01), f'Computed deflection {tip_deflection} differs from analytical {analytical_deflection}'

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
    node_coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 3.0, 0.0], [2.0, 1.0, 5.0]])
    props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0001}
    elements = [{'node_i': 0, 'node_j': 3, **props}, {'node_i': 1, 'node_j': 3, **props}, {'node_i': 2, 'node_j': 3, **props}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 1: [1, 1, 1, 1, 1, 1], 2: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0.0), 'Displacements must be zero for zero load.'
    assert np.allclose(r_zero, 0.0), 'Reactions must be zero for zero load.'
    load_vals = np.array([5000.0, -2000.0, 1000.0, 50.0, -50.0, 100.0])
    nodal_loads_base = {3: load_vals}
    (u_base, r_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads_base)
    assert not np.allclose(u_base, 0.0), 'Displacements should be non-zero.'
    assert not np.allclose(r_base, 0.0), 'Reactions should be non-zero.'
    nodal_loads_double = {3: 2.0 * load_vals}
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u_double, 2.0 * u_base, rtol=1e-05), 'Displacements did not scale linearly.'
    assert np.allclose(r_double, 2.0 * r_base, rtol=1e-05), 'Reactions did not scale linearly.'
    nodal_loads_neg = {3: -1.0 * load_vals}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u_neg, -u_base, rtol=1e-05), 'Displacements did not invert correctly.'
    assert np.allclose(r_neg, -r_base, rtol=1e-05), 'Reactions did not invert correctly.'
    sum_forces = np.zeros(3)
    sum_moments = np.zeros(3)
    for (node_idx, loads) in nodal_loads_base.items():
        F = np.array(loads[:3])
        M = np.array(loads[3:])
        pos = node_coords[node_idx]
        sum_forces += F
        sum_moments += M + np.cross(pos, F)
    n_nodes = len(node_coords)
    r_matrix = r_base.reshape((n_nodes, 6))
    for i in range(n_nodes):
        R_F = r_matrix[i, :3]
        R_M = r_matrix[i, 3:]
        pos = node_coords[i]
        sum_forces += R_F
        sum_moments += R_M + np.cross(pos, R_F)
    assert np.allclose(sum_forces, 0.0, atol=0.0001), f'Force equilibrium failed: {sum_forces}'
    assert np.allclose(sum_moments, 0.0, atol=0.0001), f'Moment equilibrium failed: {sum_moments}'