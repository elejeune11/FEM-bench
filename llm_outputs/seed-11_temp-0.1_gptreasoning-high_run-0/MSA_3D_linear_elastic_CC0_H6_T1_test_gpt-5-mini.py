def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements,
    tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    n_elems = 10
    n_nodes = n_elems + 1
    L_total = 3.0
    axis = np.array([1.0, 1.0, 1.0])
    axis = axis / np.linalg.norm(axis)
    node_coords = np.vstack([axis * (L_total * i / n_elems) for i in range(n_nodes)])
    E = 210000000000.0
    nu = 0.3
    A = 0.0001
    I = 1e-06
    J = 1e-06
    elements = []
    for i in range(n_elems):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    tip_idx = n_nodes - 1
    P_mag = 1000.0
    perp = np.array([1.0, -1.0, 0.0])
    perp = perp / np.linalg.norm(perp)
    F_tip = (P_mag * perp).tolist()
    nodal_loads = {tip_idx: [F_tip[0], F_tip[1], F_tip[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert isinstance(u, np.ndarray) and isinstance(r, np.ndarray)
    assert u.shape == (6 * n_nodes,)
    assert r.shape == (6 * n_nodes,)
    assert np.allclose(u[0:6], 0.0, atol=1e-12)
    delta_expected = P_mag * L_total ** 3 / (3.0 * E * I)
    tip_disp = u[6 * tip_idx:6 * tip_idx + 3]
    delta_numerical = float(np.linalg.norm(tip_disp))
    rel_err = abs(delta_numerical - delta_expected) / max(abs(delta_expected), 1e-12)
    assert rel_err < 0.05

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    n_nodes = node_coords.shape[0]
    E = 210000000000.0
    nu = 0.3
    A = 0.001
    I_y = 1e-06
    I_z = 1e-06
    J = 1e-06
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u0, 0.0, atol=1e-12)
    assert np.allclose(r0, 0.0, atol=1e-12)
    loads1 = {2: [100.0, 200.0, -50.0, 2.0, 3.0, -1.0], 3: [0.0, -100.0, 50.0, 5.0, 0.0, -2.0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, loads1)
    assert np.any(np.abs(u1) > 1e-12)
    assert np.any(np.abs(r1) > 1e-12)
    loads2 = {k: [2.0 * val for val in v] for (k, v) in loads1.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, loads2)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-06, atol=1e-09)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-06, atol=1e-09)
    loads_neg = {k: [-val for val in v] for (k, v) in loads1.items()}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, loads_neg)
    assert np.allclose(u_neg, -u1, rtol=1e-06, atol=1e-09)
    assert np.allclose(r_neg, -r1, rtol=1e-06, atol=1e-09)
    P_global = np.zeros(6 * n_nodes)
    for (node_idx, vals) in loads1.items():
        P_global[6 * node_idx:6 * node_idx + 6] = np.array(vals, dtype=float)
    total = np.sum((r1 + P_global).reshape((n_nodes, 6)), axis=0)
    assert np.allclose(total, np.zeros(6), atol=1e-06)