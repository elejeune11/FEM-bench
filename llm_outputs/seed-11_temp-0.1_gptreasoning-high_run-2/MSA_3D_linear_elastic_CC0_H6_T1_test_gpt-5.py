def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements,
    tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the analytical reference solution δ = F L^3 / (3 E I).
    """
    n_elems = 10
    n_nodes = n_elems + 1
    axis = np.array([1.0, 1.0, 1.0])
    axis_unit = axis / np.linalg.norm(axis)
    L = 3.0
    start = np.zeros(3)
    end = axis_unit * L
    node_coords = np.array([start + i / n_elems * (end - start) for i in range(n_nodes)], dtype=float)
    E = 210000000000.0
    nu = 0.3
    r = 0.05
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    J = 2.0 * I
    elements = []
    for i in range(n_elems):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0], dtype=float)})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_mag = 1000.0
    F_dir = np.array([1.0, -1.0, 0.0], dtype=float)
    F_dir_unit = F_dir / np.linalg.norm(F_dir)
    assert abs(np.dot(F_dir_unit, axis_unit)) < 1e-12
    tip_node = n_nodes - 1
    nodal_loads = {tip_node: [*F_mag * F_dir_unit, 0.0, 0.0, 0.0]}
    u, r = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert u.shape == (6 * n_nodes,)
    assert r.shape == (6 * n_nodes,)
    delta_analytical = F_mag * L ** 3 / (3.0 * E * I)
    u_tip = u[6 * tip_node:6 * tip_node + 3]
    delta_proj = float(np.dot(u_tip, F_dir_unit))
    assert np.isclose(delta_proj, delta_analytical, rtol=5e-06, atol=1e-10)
    perp_component = np.linalg.norm(u_tip - delta_proj * F_dir_unit)
    assert perp_component <= max(1e-06 * abs(delta_analytical), 1e-12)
    axial_component = float(np.dot(u_tip, axis_unit))
    assert abs(axial_component) <= max(1e-06 * abs(delta_analytical), 1e-12)
    r_nodes = r.reshape(n_nodes, 6)
    total_reaction_force = r_nodes[:, 0:3].sum(axis=0)
    total_applied_force = np.zeros(3)
    total_applied_force += F_mag * F_dir_unit
    assert np.allclose(total_reaction_force, -total_applied_force, rtol=1e-08, atol=1e-10)

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
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 1.0], [2.0, 1.0, 1.5], [2.0, 1.5, 2.5], [0.5, 1.2, 2.2], [3.0, -0.6, 0.5]], dtype=float)
    n_nodes = node_coords.shape[0]
    E = 70000000000.0
    nu = 0.33
    A = 0.008
    I_y = 8e-06
    I_z = 5e-06
    J = I_y + I_z
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 3, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 2, 'node_j': 5, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 5, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    loads_zero = {}
    u0, r0 = fcn(node_coords, elements, boundary_conditions, loads_zero)
    assert u0.shape == (6 * n_nodes,)
    assert r0.shape == (6 * n_nodes,)
    assert np.allclose(u0, 0.0, atol=1e-12)
    assert np.allclose(r0, 0.0, atol=1e-12)
    loads = {3: [500.0, -800.0, 200.0, 100.0, 0.0, -50.0], 4: [-200.0, 400.0, 0.0, 0.0, 80.0, 30.0], 5: [1000.0, 200.0, -600.0, -120.0, 60.0, 0.0]}
    u1, r1 = fcn(node_coords, elements, boundary_conditions, loads)
    assert np.linalg.norm(u1) > 0.0
    assert np.linalg.norm(r1) > 0.0
    loads2 = {k: [2.0 * x for x in v] for k, v in loads.items()}
    u2, r2 = fcn(node_coords, elements, boundary_conditions, loads2)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-08, atol=1e-12)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-08, atol=1e-12)
    loads_neg = {k: [-x for x in v] for k, v in loads.items()}
    u_neg, r_neg = fcn(node_coords, elements, boundary_conditions, loads_neg)
    assert np.allclose(u_neg, -u1, rtol=1e-08, atol=1e-12)
    assert np.allclose(r_neg, -r1, rtol=1e-08, atol=1e-12)
    r1_nodes = r1.reshape(n_nodes, 6)
    total_reaction_force = r1_nodes[:, 0:3].sum(axis=0)
    total_applied_force = np.zeros(3)
    for n, vec in loads.items():
        total_applied_force += np.array(vec[0:3], dtype=float)
    assert np.allclose(total_reaction_force, -total_applied_force, rtol=1e-08, atol=1e-10)
    total_reaction_moment = np.zeros(3)
    for i in range(n_nodes):
        rF = r1_nodes[i, 0:3]
        rM = r1_nodes[i, 3:6]
        x = node_coords[i]
        total_reaction_moment += rM + np.cross(x, rF)
    total_applied_moment = np.zeros(3)
    for n, vec in loads.items():
        F = np.array(vec[0:3], dtype=float)
        M = np.array(vec[3:6], dtype=float)
        x = node_coords[n]
        total_applied_moment += M + np.cross(x, F)
    assert np.allclose(total_reaction_moment, -total_applied_moment, rtol=1e-08, atol=1e-10)