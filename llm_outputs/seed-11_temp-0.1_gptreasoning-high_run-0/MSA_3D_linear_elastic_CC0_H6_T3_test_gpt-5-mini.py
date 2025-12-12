def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    n_elems = 10
    L_total = 1.0
    le = L_total / n_elems
    axis = np.array([1.0, 1.0, 1.0], dtype=float)
    axis_unit = axis / np.linalg.norm(axis)
    node_coords = np.array([axis_unit * (i * le) for i in range(n_elems + 1)], dtype=float)
    E = 210000000000.0
    nu = 0.3
    A = 1.0
    I = 1e-06
    J = 1e-06
    elements = []
    for i in range(n_elems):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_mag = 1000.0
    f_dir = np.array([-1.0, 1.0, 0.0], dtype=float)
    f_dir /= np.linalg.norm(f_dir)
    tip_node = n_elems
    nodal_loads = {tip_node: [float(F_mag * f_dir[0]), float(F_mag * f_dir[1]), float(F_mag * f_dir[2]), 0.0, 0.0, 0.0]}
    (u, r) = fcn(np.asarray(node_coords, dtype=float), elements, boundary_conditions, nodal_loads)
    N_nodes = n_elems + 1
    assert isinstance(u, np.ndarray)
    assert isinstance(r, np.ndarray)
    assert u.shape == (6 * N_nodes,)
    assert r.shape == (6 * N_nodes,)
    assert np.allclose(u[0:6], 0.0, atol=1e-12)
    delta = F_mag * L_total ** 3 / (3.0 * E * I)
    expected_tip_disp = delta * f_dir
    u_tip = u[tip_node * 6:tip_node * 6 + 3]
    assert np.allclose(u_tip, expected_tip_disp, rtol=0.03, atol=1e-08)
    applied_forces = np.zeros((N_nodes, 3), dtype=float)
    for (ni, load) in nodal_loads.items():
        applied_forces[ni, :] = load[0:3]
    reaction = r.reshape((N_nodes, 6))
    reaction_forces = reaction[:, 0:3]
    total_force_balance = applied_forces.sum(axis=0) + reaction_forces.sum(axis=0)
    assert np.allclose(total_force_balance, 0.0, atol=1e-06)
    applied_moments = np.zeros((N_nodes, 3), dtype=float)
    for (ni, load) in nodal_loads.items():
        applied_moments[ni, :] = load[3:6]
    reaction_moments = reaction[:, 3:6]
    pos = node_coords
    total_moment_balance = applied_moments.sum(axis=0) + reaction_moments.sum(axis=0) + np.cross(pos, applied_forces).sum(axis=0) + np.cross(pos, reaction_forces).sum(axis=0)
    assert np.allclose(total_moment_balance, 0.0, atol=1e-05)

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
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    N_nodes = node_coords.shape[0]
    E = 210000000000.0
    nu = 0.3
    A = 0.001
    I_y = 2e-06
    I_z = 2e-06
    J = 1e-06
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 0, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 0, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    zero_loads = {}
    (u0, r0) = fcn(np.asarray(node_coords, dtype=float), elements, boundary_conditions, zero_loads)
    assert isinstance(u0, np.ndarray) and isinstance(r0, np.ndarray)
    assert u0.shape == (6 * N_nodes,)
    assert r0.shape == (6 * N_nodes,)
    assert np.allclose(u0, 0.0, atol=1e-12)
    assert np.allclose(r0, 0.0, atol=1e-12)
    loads = {1: [100.0, -50.0, 25.0, 10.0, -5.0, 2.5], 2: [-80.0, 60.0, 0.0, 0.0, 7.0, -3.0], 3: [0.0, -100.0, 200.0, -6.0, 0.0, 4.0]}
    (u1, r1) = fcn(np.asarray(node_coords, dtype=float), elements, boundary_conditions, loads)
    assert np.linalg.norm(u1) > 1e-12
    assert np.linalg.norm(r1) > 1e-12
    loads_double = {k: [2.0 * float(x) for x in v] for (k, v) in loads.items()}
    (u2, r2) = fcn(np.asarray(node_coords, dtype=float), elements, boundary_conditions, loads_double)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-06, atol=1e-12)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-06, atol=1e-12)
    loads_neg = {k: [-float(x) for x in v] for (k, v) in loads.items()}
    (u_neg, r_neg) = fcn(np.asarray(node_coords, dtype=float), elements, boundary_conditions, loads_neg)
    assert np.allclose(u_neg, -u1, rtol=1e-06, atol=1e-12)
    assert np.allclose(r_neg, -r1, rtol=1e-06, atol=1e-12)
    applied_forces = np.zeros((N_nodes, 3), dtype=float)
    applied_moments = np.zeros((N_nodes, 3), dtype=float)
    for (ni, load) in loads.items():
        applied_forces[ni, :] = np.asarray(load[0:3], dtype=float)
        applied_moments[ni, :] = np.asarray(load[3:6], dtype=float)
    reaction = r1.reshape((N_nodes, 6))
    reaction_forces = reaction[:, 0:3]
    reaction_moments = reaction[:, 3:6]
    total_force_balance = applied_forces.sum(axis=0) + reaction_forces.sum(axis=0)
    assert np.allclose(total_force_balance, 0.0, atol=1e-06)
    pos = node_coords
    total_moment_balance = applied_moments.sum(axis=0) + reaction_moments.sum(axis=0) + np.cross(pos, applied_forces).sum(axis=0) + np.cross(pos, reaction_forces).sum(axis=0)
    assert np.allclose(total_moment_balance, 0.0, atol=1e-05)